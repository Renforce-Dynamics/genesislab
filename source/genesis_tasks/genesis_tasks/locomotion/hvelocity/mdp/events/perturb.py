"""Perturbation-related event functions (pushes, external wrenches)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch

from genesislab.managers import SceneEntityCfg

from .utils import resolve_env_ids, sample_range, sample_range_dict

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


def push_by_setting_velocity(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    velocity_range: Dict[str, Tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply a random push by modifying the root velocity within given ranges."""
    env_ids = resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    entity_name = asset_cfg.entity_name
    robot = env.scene.entities[entity_name]
    lin_vel_w, ang_vel_w = robot.data.root_lin_vel_w, robot.data.root_ang_vel_w
    lin_vel_w = lin_vel_w[env_ids].clone()
    ang_vel_w = ang_vel_w[env_ids].clone()

    device = env.device
    num_envs = env_ids.numel()

    vel_offsets = sample_range_dict(
        velocity_range,
        keys=("x", "y", "z", "roll", "pitch", "yaw"),
        num_envs=num_envs,
        device=device,
    )
    lin_offsets = vel_offsets[:, :3]
    ang_offsets = vel_offsets[:, 3:]

    lin_vel_w = lin_vel_w + lin_offsets
    ang_vel_w = ang_vel_w + ang_offsets

    env.scene.controller.set_root_state(
        entity_name=entity_name,
        linear_velocity=lin_vel_w,
        angular_velocity=ang_vel_w,
        env_ids=env_ids,
    )


def apply_external_force_torque(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Approximate external pushes by sampling equivalent velocity perturbations."""
    env_ids = resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    if force_range[0] == force_range[1] == 0.0 and torque_range[0] == torque_range[1] == 0.0:
        return

    entity_name = asset_cfg.entity_name
    _, _, lin_vel_w, ang_vel_w = env.scene.querier.get_root_state(entity_name)
    lin_vel_w = lin_vel_w[env_ids].clone()
    ang_vel_w = ang_vel_w[env_ids].clone()

    device = env.device
    num_envs = env_ids.numel()

    lin_offsets = sample_range(force_range[0], force_range[1], (num_envs, 3), device=device)
    ang_offsets = sample_range(torque_range[0], torque_range[1], (num_envs, 3), device=device)

    lin_vel_w = lin_vel_w + lin_offsets
    ang_vel_w = ang_vel_w + ang_offsets

    env.scene.controller.set_root_state(
        entity_name=entity_name,
        linear_velocity=lin_vel_w,
        angular_velocity=ang_vel_w,
        env_ids=env_ids,
    )


__all__ = [
    "push_by_setting_velocity",
    "apply_external_force_torque",
]

