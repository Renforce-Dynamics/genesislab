"""Common reward functions for velocity tracking locomotion tasks.

These functions can be used to define reward terms in the MDP configuration.
They follow the same interface as IsaacLab's reward functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesislab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


"""
Root penalties.
"""


def lin_vel_z_l2(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    entity = env.entities[asset_cfg.entity_name]
    # Use body frame velocity if available, otherwise world frame
    lin_vel = entity.data.root_lin_vel_b if hasattr(entity.data, "root_lin_vel_b") else entity.data.root_lin_vel_w
    return torch.square(lin_vel[:, 2])


def ang_vel_xy_l2(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    entity = env.entities[asset_cfg.entity_name]
    # Use body frame velocity if available, otherwise world frame
    ang_vel = entity.data.root_ang_vel_b if hasattr(entity.data, "root_ang_vel_b") else entity.data.root_ang_vel_w
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def flat_orientation_l2(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    entity = env.entities[asset_cfg.entity_name]
    projected_gravity = entity.data.projected_gravity_b
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


"""
Joint penalties.
"""


def joint_torques_l2(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    entity = env.entities[asset_cfg.entity_name]
    # TODO: Get applied torques from entity data when available
    # For now, return zeros as placeholder
    # applied_torque = entity.data.applied_torque
    # if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
    #     return torch.sum(torch.square(applied_torque[:, asset_cfg.joint_ids]), dim=1)
    # return torch.sum(torch.square(applied_torque), dim=1)
    return torch.zeros(env.num_envs, device=env.device)


def joint_acc_l2(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    entity = env.entities[asset_cfg.entity_name]
    # TODO: Get joint accelerations from entity data when available
    # For now, compute from velocity differences
    joint_vel = entity.data.joint_vel
    # Simple approximation: use velocity as proxy (not ideal but works as placeholder)
    # In practice, this should be computed from finite differences of joint_vel
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        return torch.sum(torch.square(joint_vel[:, asset_cfg.joint_ids]), dim=1) * 0.01  # Scale down
    return torch.sum(torch.square(joint_vel), dim=1) * 0.01


def joint_pos_limits(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    entity = env.entities[asset_cfg.entity_name]
    joint_pos = entity.data.joint_pos
    
    # Get soft limits if available
    if hasattr(entity.data, "soft_joint_pos_limits"):
        soft_limits = entity.data.soft_joint_pos_limits
        if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
            joint_pos = joint_pos[:, asset_cfg.joint_ids]
            soft_limits = soft_limits[:, asset_cfg.joint_ids]
        
        # Compute out of limits violations
        out_of_limits = -(joint_pos - soft_limits[:, :, 0]).clip(max=0.0)
        out_of_limits += (joint_pos - soft_limits[:, :, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)
    else:
        # No limits available, return zeros
        return torch.zeros(env.num_envs, device=env.device)


"""
Action penalties.
"""


def action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel.

    Args:
        env: The environment instance.

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    if hasattr(env.action_manager, "prev_action"):
        return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    else:
        # No previous action available, return zeros
        return torch.zeros(env.num_envs, device=env.device)


"""
Contact sensor penalties.
"""


def undesired_contacts(env: "ManagerBasedRlEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold.

    Args:
        env: The environment instance.
        threshold: Force threshold for contact detection.
        sensor_cfg: Configuration for the contact sensor.

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    # TODO: Implement when contact sensor is available
    # For now, return zeros as placeholder
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_contact_forces = contact_sensor.data.net_forces_w_history
    # is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # return torch.sum(is_contact, dim=1)
    return torch.zeros(env.num_envs, device=env.device)


"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
    env: "ManagerBasedRlEnv", std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel.

    Args:
        env: The environment instance.
        std: Standard deviation for the exponential kernel.
        command_name: Name of the command term.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the reward.
    """
    entity = env.entities[asset_cfg.entity_name]
    command = env.command_manager.get_command(command_name)
    
    # Get body frame velocity if available, otherwise world frame
    lin_vel = entity.data.root_lin_vel_b if hasattr(entity.data, "root_lin_vel_b") else entity.data.root_lin_vel_w
    
    # Compute error in xy plane
    lin_vel_error = torch.sum(torch.square(command[:, :2] - lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: "ManagerBasedRlEnv", std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel.

    Args:
        env: The environment instance.
        std: Standard deviation for the exponential kernel.
        command_name: Name of the command term.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the reward.
    """
    entity = env.entities[asset_cfg.entity_name]
    command = env.command_manager.get_command(command_name)
    
    # Get body frame velocity if available, otherwise world frame
    ang_vel = entity.data.root_ang_vel_b if hasattr(entity.data, "root_ang_vel_b") else entity.data.root_ang_vel_w
    
    # Compute error in z (yaw) component
    ang_vel_error = torch.square(command[:, 2] - ang_vel[:, 2])
    return torch.exp(-ang_vel_error / std**2)


"""
Task-specific rewards (from velocity task).
"""


def feet_air_time(
    env: "ManagerBasedRlEnv", command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold.
    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.

    Args:
        env: The environment instance.
        command_name: Name of the command term.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Minimum air time threshold.

    Returns:
        Tensor of shape (num_envs,) containing the reward.
    """
    # TODO: Implement when contact sensor is available
    # For now, return zeros as placeholder
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    # return reward
    return torch.zeros(env.num_envs, device=env.device)
