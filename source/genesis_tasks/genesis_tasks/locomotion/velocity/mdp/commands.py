"""Common command terms for velocity tracking locomotion tasks.

These command terms can be used to define commands in the MDP configuration.
They follow the same interface as mjlab/IsaacLab's command terms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from genesislab.managers.command_manager import CommandTerm, CommandTermCfg
from genesislab.utils.configclass import configclass
from genesislab.components.markers.arrow_markers import ArrowMarkers, ArrowMarkersCfg

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator.

    This command generator samples velocity commands uniformly from specified ranges.
    """

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False."""

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command.
    Defaults to 1.0. Only used if heading_command is True.
    """

    init_velocity_prob: float = 0.0
    """Probability of initializing environments with the sampled velocity command. Defaults to 0.0."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.
        Only used if heading_command is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    @configclass
    class VizCfg:
        """Visualization configuration for debug arrows."""

        z_offset: float = 0.2
        """Z-offset above the robot base for drawing arrows. Defaults to 0.2."""

        scale: float = 0.5
        """Scale factor for arrow lengths. Defaults to 0.5."""

    viz: VizCfg = VizCfg()
    """Visualization configuration for debug arrows."""

    def build(self, env: "ManagerBasedRlEnv") -> "UniformVelocityCommand":
        """Build the uniform velocity command term from this config."""
        return UniformVelocityCommand(cfg=self, env=env)

    def __post_init__(self):
        """Validate configuration."""
        if self.heading_command and self.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but "
                "the `ranges.heading` parameter is set to None."
            )
        if self.ranges.heading is not None and not self.heading_command:
            raise ValueError("ranges.heading is set but heading_command=False.")


class UniformVelocityCommand(CommandTerm):
    """Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.
    """

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: "ManagerBasedRlEnv"):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)

        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError("heading_command=True but ranges.heading is set to None.")
        if self.cfg.ranges.heading is not None and not self.cfg.heading_command:
            raise ValueError("ranges.heading is set but heading_command=False.")

        self.robot = env.entities[cfg.asset_name]

        # Create buffers to store the command
        # Command: [x vel, y vel, yaw vel]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        # Metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self) -> None:
        """Update metrics based on current state."""
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # Get body frame velocities (prefer body frame if available)
        lin_vel_b = (
            self.robot.data.root_lin_vel_b
            if hasattr(self.robot.data, "root_lin_vel_b")
            else self.robot.data.root_lin_vel_w
        )
        ang_vel_b = (
            self.robot.data.root_ang_vel_b
            if hasattr(self.robot.data, "root_ang_vel_b")
            else self.robot.data.root_ang_vel_w
        )

        # Update metrics
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Resample velocity command for specified environments.

        Args:
            env_ids: Environment indices to resample.
        """
        if len(env_ids) == 0:
            return

        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.cfg.heading_command:
            assert self.cfg.ranges.heading is not None
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # Optionally initialize environments with the sampled velocity
        init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
        init_vel_env_ids = env_ids[init_vel_mask]
        if len(init_vel_env_ids) > 0:
            # Get current root state
            root_pos = self.robot.data.root_pos_w[init_vel_env_ids]
            root_quat = self.robot.data.root_quat_w[init_vel_env_ids]

            # Set linear velocity in body frame
            lin_vel_b = (
                self.robot.data.root_lin_vel_b[init_vel_env_ids]
                if hasattr(self.robot.data, "root_lin_vel_b")
                else torch.zeros((len(init_vel_env_ids), 3), device=self.device)
            )
            lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]

            # Transform to world frame
            # Simplified: assume quat is [x, y, z, w] format
            if root_quat.shape[-1] == 4:
                # Convert quaternion to rotation matrix (simplified quaternion rotation)
                # quat format: [x, y, z, w] from Genesis
                # Normalize
                quat_norm = root_quat / torch.norm(root_quat, dim=-1, keepdim=True)
                qx, qy, qz, qw = quat_norm[:, 0], quat_norm[:, 1], quat_norm[:, 2], quat_norm[:, 3]
                # Convert to [w, x, y, z] for rotation
                xyz = torch.stack([qx, qy, qz], dim=-1)  # (num_envs, 3)
                w = qw.unsqueeze(-1)  # (num_envs, 1)
                # quat_apply: v' = v + 2*w*cross(xyz, v) + 2*cross(xyz, cross(xyz, v))
                t = xyz.cross(lin_vel_b, dim=-1) * 2
                lin_vel_w = lin_vel_b + w * t + xyz.cross(t, dim=-1)
            else:
                lin_vel_w = lin_vel_b

            # Set angular velocity in body frame
            ang_vel_b = (
                self.robot.data.root_ang_vel_b[init_vel_env_ids]
                if hasattr(self.robot.data, "root_ang_vel_b")
                else torch.zeros((len(init_vel_env_ids), 3), device=self.device)
            )
            ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]

            # Write root state to simulation
            # Note: This requires access to the underlying entity's write_root_state method
            # For now, this is a placeholder that would need to be implemented in the entity layer
            # self.robot.write_root_state_to_sim(
            #     torch.cat([root_pos, root_quat, lin_vel_w, ang_vel_b], dim=-1),
            #     init_vel_env_ids,
            # )

    def _update_command(self) -> None:
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        if self.cfg.heading_command:
            # Compute heading from quaternion
            quat = self.robot.data.root_quat_w
            # Extract yaw from quaternion (simplified)
            if quat.shape[-1] == 4:
                # Assuming [x, y, z, w] format
                yaw_current = torch.atan2(
                    2 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]),
                    1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
                )
            else:
                yaw_current = torch.zeros(self.num_envs, device=self.device)

            # Compute heading error and wrap to [-pi, pi]
            heading_error = self.heading_target - yaw_current
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

            # Update angular velocity for heading environments
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.vel_command_b[env_ids, 2] = torch.clamp(
                    self.cfg.heading_control_stiffness * heading_error[env_ids],
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )

        # Set velocity to zero for standing environments
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0

    def _debug_vis_impl(self, visualizer: Any) -> None:
        """Batch debug visualization of velocity commands (IsaacLab-style)."""
        import numpy as np
        import torch

        # Lazy-create marker groups (one for desired, one for actual velocity).
        if not hasattr(self, "_goal_vel_markers"):
            scene = self._env._binding.scene
            self._goal_vel_markers = ArrowMarkers(
                scene, ArrowMarkersCfg(radius=0.015, color=(0.2, 0.9, 0.2, 0.9))
            )
            self._current_vel_markers = ArrowMarkers(
                scene, ArrowMarkersCfg(radius=0.015, color=(0.0, 0.7, 1.0, 0.9))
            )

        # Current base positions in world frame and lift them by z_offset.
        base_pos_w = self.robot.data.link_pos_w[:, 1].clone()  # (N, 3)
        base_pos_w[:, 2] += self.cfg.viz.z_offset

        # Desired and actual XY velocities in base frame.
        cmd_xy_b = self.command[:, :2]  # (N, 2)
        lin_vel_xy_b = (
            self.robot.data.root_lin_vel_b[:, :2]
            if hasattr(self.robot.data, "root_lin_vel_b")
            else self.robot.data.root_lin_vel_w[:, :2]
        )

        device = self.device
        scale = self.cfg.viz.scale

        # Helper: rotate a batch of base-frame XY vectors into world frame using quaternions.
        def _rotate_xy_to_world(xy_b: torch.Tensor) -> torch.Tensor:
            # xy_b: (N, 2) -> vec_b: (N, 3)
            vec_b = torch.zeros(xy_b.shape[0], 3, device=device)
            vec_b[:, :2] = xy_b
            quat = self.robot.data.root_quat_w  # (N, 4) assumed [x, y, z, w]
            # Normalize quaternions
            quat = quat / torch.norm(quat, dim=-1, keepdim=True)
            qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            xyz = torch.stack([qx, qy, qz], dim=-1)  # (N, 3)
            w = qw.unsqueeze(-1)  # (N, 1)
            # Quaternion apply: v' = v + 2*w*cross(xyz, v) + 2*cross(xyz, cross(xyz, v))
            t = xyz.cross(vec_b, dim=-1) * 2.0
            vec_w = vec_b + w * t + xyz.cross(t, dim=-1)
            return vec_w

        # Rotate desired and actual velocities to world frame and scale.
        cmd_vec_w = _rotate_xy_to_world(cmd_xy_b.to(device)) * scale
        act_vec_w = _rotate_xy_to_world(lin_vel_xy_b.to(device)) * scale

        # Filter out uninitialized robots (at origin) to avoid clutter.
        base_pos_w_np = base_pos_w.detach().cpu().numpy()
        cmd_vec_np = cmd_vec_w.detach().cpu().numpy()
        act_vec_np = act_vec_w.detach().cpu().numpy()

        valid_mask = np.linalg.norm(base_pos_w_np, axis=1) > 1e-6

        # Visualize desired and actual velocities via marker groups (batched).
        self._goal_vel_markers.visualize(
            translations=base_pos_w_np,
            directions=cmd_vec_np,
            mask=valid_mask,
        )
        self._current_vel_markers.visualize(
            translations=base_pos_w_np,
            directions=act_vec_np,
            mask=valid_mask,
        )