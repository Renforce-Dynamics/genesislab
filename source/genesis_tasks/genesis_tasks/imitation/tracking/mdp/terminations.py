from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from genesislab.managers import SceneEntityCfg

import genesislab.utils.math as math_utils  # quat_apply_inverse: wxyz
from .rewards import _get_body_indexes

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv
    from .commands import MotionCommand


def bad_anchor_pos(env: "ManagerBasedRlEnv", command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: "ManagerBasedRlEnv", command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    # Use LabEntity wrapper from GenesisLab scene entities
    asset = env.entities[asset_cfg.entity_name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: "ManagerBasedRlEnv", command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    # Directly compute position error between motion bodies and robot bodies in world frame.
    error = torch.norm(
        command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes],
        dim=-1,
    )
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: "ManagerBasedRlEnv", command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    # Same as above but only on the z-component of the position error.
    error = torch.abs(command.body_pos_w - command.robot_body_pos_w)[:, body_indexes, -1]
    return torch.any(error > threshold, dim=-1)
