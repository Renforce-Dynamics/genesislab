"""Replay motion CSVs through Genesis FK and export NPZ for imitation tracking.

Uses Genesis (gs) native scene and entities only (no LabScene). Inspired by
``.references/Genesis/examples/rendering/follow_entity.py`` and
``.references/beyondMimic/scripts/data_replay/csv_to_npz.py``.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable

from PIL import Image
import numpy as np
import torch
import tqdm

import genesis as gs

from genesislab.utils.math import quat_inv, quat_mul, quat_to_euler_xyz, quat_xyzw_to_wxyz

# Use the same G1 asset as the training env (BeyondMimic USD) so that
# link ordering and joint names match MotionCommand expectations.
from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG

# G1_USD_PATH = G1_BEYONDMIMIC_CFG.morph_path

# Joint order in the original BeyondMimic csv_to_npz pipeline (IsaacLab version).
# The motion DOFs in the CSV are in this order and must be mapped to the robot's
# internal DOF ordering manually.
G1_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay motion from CSV files through Genesis FK and export NPZ (imitation tracking)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/datasets",
        help="Directory containing input motion CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/datasets",
        help="Directory to write NPZ motion files.",
    )
    parser.add_argument(
        "--input-fps",
        type=int,
        default=30,
        help="FPS of the input motion data.",
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=50,
        help="FPS of the output motion / FK replay.",
    )
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=(
            "Optional frame range: START END (both inclusive). Frame index starts from 1. "
            "If not provided, all frames are loaded."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string for tensors (e.g. 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        help="Genesis backend ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        default=False,
        help="If set, open a Genesis viewer window while replaying FK.",
    )
    return parser


def _iter_motion_files(input_dir: str) -> Iterable[str]:
    """Yield CSV basenames (without extension) in input_dir."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith(".csv"):
            yield os.path.splitext(fname)[0]


class MotionLoader:
    """Interpolate CSV motion to target FPS and compute root/joint velocities."""

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self) -> None:
        """Load motion from CSV."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            start, end = self.frame_range
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=start - 1,
                    max_rows=end - start + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        # Layout: [base_pos(3), base_quat from CSV as legacy (x,y,z,w), joint_pos(...)]
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = quat_xyzw_to_wxyz(motion[:, 3:7])
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"[csv_to_npz] Motion loaded ({self.motion_file}), "
            f"duration: {self.duration:.3f} s, frames: {self.input_frames}"
        )

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation; unit quaternions in [w, x, y, z] (wxyz)."""
        a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        b = torch.where(dot < 0.0, -b, b)
        dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta).clamp_min(1e-8)
        t = blend.view(-1, 1)
        w1 = torch.sin((1.0 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta
        return w1 * a + w2 * b

    def _interpolate_motion(self) -> None:
        """Interpolate motion to output FPS."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"[csv_to_npz] Motion interpolated: "
            f"in={self.input_frames}@{self.input_fps}Hz -> "
            f"out={self.output_frames}@{self.output_fps}Hz"
        )

    def _compute_velocities(self) -> None:
        """Compute root linear, angular and joint velocities with finite differences."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]

        q = self.motion_base_rots
        q_prev, q_next = q[:-2], q[2:]
        # Relative rotation q_next * inv(q_prev); quats are wxyz.
        q_rel = quat_mul(q_next, quat_inv(q_prev))
        q_rel = q_rel / q_rel.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        angle = 2.0 * torch.acos(q_rel[..., 0].clamp(-1.0, 1.0))
        sin_half = torch.sin(angle / 2.0).clamp_min(1e-8)
        axis = q_rel[..., :3] / sin_half.unsqueeze(-1)
        omega = (axis * angle.unsqueeze(-1)) / (2.0 * self.output_dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        self.motion_base_ang_vels = omega

    def get_next_state(self):
        """Return next state and whether we've wrapped around."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def build_gs_scene(
    output_fps: int,
    show_viewer: bool,
) -> tuple[gs.Scene, any]:
    """Create a Genesis scene with plane + G1 USD, build it, return (scene, robot_entity)."""
    dt = 1.0 / float(output_fps)
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(dt=dt),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    robot_entity = scene.add_entity(
        gs.morphs.MJCF(
            file=G1_FULL_ACT_CFG.morph_path,
            pos=(0.0, 0.0, 0.76),
            quat=(0.0, 0.0, 0.0, 1.0),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Copper(
            color=(0.4, 0.7, 1.0),
            opacity=0.5,
        ),
        name="robot",
        vis_mode="visual"
    )
    scene.build(n_envs=1, env_spacing=(2.0, 2.0))
    return scene, robot_entity


def run_fk_for_motion(
    scene: gs.Scene,
    robot_entity: "gs.engine.entities.KinematicEntity",
    motion: MotionLoader,
    device: torch.device,
    show_window: bool = False,
) -> dict[str, np.ndarray]:
    """Run FK for a single motion sequence; return logged arrays.

    Dataset provides base orientation as quat; engine uses Euler (XYZ) for base orientation.
    Base position DOFs = 6 (pos 3 + euler 3), base velocity DOFs = 6 (lin 3 + ang 3).
    When show_window is True, calls scene.step() each frame so the viewer updates
    (throttled by output_fps).
    """
    dofs0 = robot_entity.get_dofs_position()
    n_envs, n_dofs = dofs0.shape
    base_pos_dofs = 6   # pos(3) + euler_xyz(3)
    base_vel_dofs = 6   # lin_vel(3) + ang_vel(3)

    # Map dataset DOF order (G1_JOINT_NAMES) to the robot's internal DOF indices.
    # This is critical so that FK uses the correct joint for each motion DOF.
    joint_dof_indices = [robot_entity.get_joint(name).dofs_idx_local[0] for name in G1_JOINT_NAMES]
    num_mapped_joints = len(joint_dof_indices)

    log = {
        "fps": [motion.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_rot_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for _ in tqdm.tqdm(
        range(motion.output_frames),
        desc="FK frames",
        unit="frame",
        leave=False,
    ):
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ), _ = motion.get_next_state()

        # Dataset root quat is wxyz after load; engine uses Euler XYZ for base orientation.
        motion_base_euler = quat_to_euler_xyz(motion_base_rot)

        dofs_pos = robot_entity.get_dofs_position()
        dofs_vel = robot_entity.get_dofs_velocity()
        dofs_pos[:, 0:3] = motion_base_pos
        dofs_pos[:, 3:6] = motion_base_euler
        dofs_pos[:, joint_dof_indices] = motion_dof_pos[:, :num_mapped_joints]
        dofs_vel[:, 0:3] = motion_base_lin_vel
        dofs_vel[:, 3:6] = motion_base_ang_vel
        dofs_vel[:, joint_dof_indices] = motion_dof_vel[:, :num_mapped_joints]

        robot_entity.set_dofs_position(dofs_pos)
        robot_entity.set_dofs_velocity(dofs_vel)

        if show_window:
            scene.step()
            time.sleep(1.0 / motion.output_fps)

        joint_pos_full = robot_entity.get_dofs_position()[:, 6:]
        joint_vel_full = robot_entity.get_dofs_velocity()[:, 6:]
        link_rot = robot_entity.get_dofs_position()[:, 3:6]
        link_pos = robot_entity.get_links_pos()
        link_quat = robot_entity.get_links_quat()
        link_lin_vel = robot_entity.get_links_vel()
        link_ang_vel = robot_entity.get_links_ang()

        log["joint_pos"].append(joint_pos_full[0].cpu().numpy().copy())
        log["joint_vel"].append(joint_vel_full[0].cpu().numpy().copy())
        log["body_pos_w"].append(link_pos[0].cpu().numpy().copy())
        log["body_quat_w"].append(link_quat[0].cpu().numpy().copy())
        log["body_rot_w"].append(link_rot[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(link_lin_vel[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(link_ang_vel[0].cpu().numpy().copy())

    for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_rot_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[k] = np.stack(log[k], axis=0)

    return log


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    gs.init()

    device = torch.device(args.device)

    scene, robot_entity = build_gs_scene(
        output_fps=args.output_fps,
        show_viewer=args.window,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    motion_basenames = list(_iter_motion_files(args.input_dir))
    for basename in tqdm.tqdm(motion_basenames, desc="Motions", unit="file"):
        csv_path = os.path.join(args.input_dir, f"{basename}.csv")
        npz_path = os.path.join(args.output_dir, f"{basename}.npz")
        tqdm.tqdm.write(f"[csv_to_npz] {basename}.csv -> {npz_path}")

        frame_range = None
        if args.frame_range is not None:
            frame_range = (args.frame_range[0], args.frame_range[1])

        motion = MotionLoader(
            motion_file=csv_path,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            device=device,
            frame_range=frame_range,
        )

        log = run_fk_for_motion(
            scene, robot_entity, motion, device=device, show_window=args.window
        )
        np.savez(npz_path, **log)


if __name__ == "__main__":
    main()
