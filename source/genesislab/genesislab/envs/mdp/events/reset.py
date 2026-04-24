"""Reset-related event functions for velocity locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesislab.managers import SceneEntityCfg
from genesislab.utils.math import quat_from_euler_xyz, quat_mul

from .utils import (
    resolve_env_ids,
    sample_range,
    sample_range_dict,
)

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


_POSE_KEYS: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw")
_VEL_KEYS: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw")


def reset_root_state_uniform(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the asset root state to a random pose and velocity within given ranges.

    ``pose_range`` / ``velocity_range`` accept keys from
    ``{"x", "y", "z", "roll", "pitch", "yaw"}``; any other key raises ``KeyError``
    (previously silently ignored, which hid typos like ``"Z"`` vs ``"z"``).

    The returned root quaternion is wxyz (Genesis / MuJoCo convention) and is
    re-normalized after composing with the sampled rotation offset, so the
    downstream controller always receives a unit quaternion.
    """
    env_ids = resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot = env.scene.entities[asset_cfg.entity_name]
    # Use default root state as the baseline, so repeated resets do not drift.
    pos_w = robot.data.default_root_pos_w[env_ids].clone()
    quat_w = robot.data.default_root_quat_w[env_ids].clone()
    lin_vel_w = robot.data.default_root_lin_vel_w[env_ids].clone()
    ang_vel_w = robot.data.default_root_ang_vel_w[env_ids].clone()
    device = env.device
    num_envs = env_ids.numel()
    pose_offsets = sample_range_dict(
        pose_range,
        keys=_POSE_KEYS,
        num_envs=num_envs,
        device=device,
    )
    pos_offsets = pose_offsets[:, :3]
    rot_offsets = pose_offsets[:, 3:]  # roll, pitch, yaw
    pos_w = pos_w + pos_offsets

    # Always compose the offset rotation. ``quat_mul`` in rotation.py already normalizes
    # its inputs so the identity offset is a no-op up to FP eps; guarding on
    # ``abs().sum() > 0`` was fragile around FP near-zero and hid bugs when ranges were
    # vanishingly small but non-zero.
    roll, pitch, yaw = rot_offsets.unbind(-1)
    quat_offset = quat_from_euler_xyz(roll, pitch, yaw)  # wxyz
    quat_w = quat_mul(quat_w, quat_offset)  # quat_mul normalizes inputs; output is unit.

    vel_offsets = sample_range_dict(
        velocity_range,
        keys=_VEL_KEYS,
        num_envs=num_envs,
        device=device,
    )
    lin_offsets = vel_offsets[:, :3]
    ang_offsets = vel_offsets[:, 3:]

    lin_vel_w = lin_vel_w + lin_offsets
    ang_vel_w = ang_vel_w + ang_offsets

    env.scene.controller.set_root_state(
        entity_name=asset_cfg.entity_name,
        position=pos_w,
        quaternion=quat_w,
        linear_velocity=lin_vel_w,
        angular_velocity=ang_vel_w,
        env_ids=env_ids,
    )


def reset_joints_by_scale(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset joint state by **multiplicative** scaling of defaults.

    For each DOF, samples ``scale ~ U(position_range)`` and sets
    ``joint_pos = default_pos * scale`` (same for velocities). Use a range **around 1.0**
    (e.g. ``(0.85, 1.15)`` or ``(0.5, 1.5)``). Ranges like ``(-1, 1)`` are invalid here: they
    crush targets toward zero and flip signs, which collapses the pose and triggers immediate
    fall / orientation terminations once gravity / body-frame velocities are computed correctly.
    """
    env_ids = resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    # Entity wrapper and data view
    lab_entity = env.entities[asset_cfg.entity_name]
    data = lab_entity.data

    default_pos = data.default_joint_pos[env_ids].clone()
    default_vel = data.default_joint_vel[env_ids].clone()

    device = env.device

    pos_scale = sample_range(position_range[0], position_range[1], default_pos.shape, device=device)
    vel_scale = sample_range(velocity_range[0], velocity_range[1], default_vel.shape, device=device)

    joint_pos = default_pos * pos_scale
    joint_vel = default_vel * vel_scale

    env.scene.controller.set_joint_positions(asset_cfg.entity_name, joint_pos, env_ids=env_ids)
    env.scene.controller.set_joint_velocities(asset_cfg.entity_name, joint_vel, env_ids=env_ids)


__all__ = [
    "reset_root_state_uniform",
    "reset_joints_by_scale",
]

