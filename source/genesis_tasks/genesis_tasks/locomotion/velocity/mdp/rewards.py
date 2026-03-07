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
    
    # Get applied torques from entity data
    if not hasattr(entity.data, "applied_torque"):
        raise AttributeError(
            f"Entity '{asset_cfg.entity_name}' data does not have 'applied_torque' attribute. "
            f"This reward term requires applied torque data from the entity."
        )
    
    applied_torque = entity.data.applied_torque
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        return torch.sum(torch.square(applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return torch.sum(torch.square(applied_torque), dim=1)


def joint_acc_l2(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    entity = env.entities[asset_cfg.entity_name]
    
    # Get joint accelerations from entity data
    if not hasattr(entity.data, "joint_acc"):
        raise AttributeError(
            f"Entity '{asset_cfg.entity_name}' data does not have 'joint_acc' attribute. "
            f"This reward term requires joint acceleration data from the entity."
        )
    
    joint_acc = entity.data.joint_acc
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        return torch.sum(torch.square(joint_acc[:, asset_cfg.joint_ids]), dim=1)
    return torch.sum(torch.square(joint_acc), dim=1)


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
    
    # Get soft limits - required for this reward term
    if not hasattr(entity.data, "soft_joint_pos_limits"):
        raise AttributeError(
            f"Entity '{asset_cfg.entity_name}' data does not have 'soft_joint_pos_limits' attribute. "
            f"This reward term requires soft joint position limits from the entity."
        )
    
    soft_limits = entity.data.soft_joint_pos_limits
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        joint_pos = joint_pos[:, asset_cfg.joint_ids]
        soft_limits = soft_limits[:, asset_cfg.joint_ids]
    
    # Compute out of limits violations
    out_of_limits = -(joint_pos - soft_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


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
    if not hasattr(env.action_manager, "prev_action"):
        raise AttributeError(
            "ActionManager does not have 'prev_action' attribute. "
            "This reward term requires the action manager to track previous actions."
        )
    
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


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
    # Look up the contact sensor from the scene.
    if not hasattr(env.scene, "sensors"):
        raise AttributeError(
            "Scene does not have 'sensors' attribute. "
            "Contact sensor must be configured in the scene."
        )
    
    if isinstance(sensor_cfg, str):
        sensor_name = sensor_cfg
    else:
        sensor_name = getattr(sensor_cfg, "entity_name", None) or getattr(sensor_cfg, "name", None) or "contact_forces"
    
    if sensor_name not in env.scene.sensors:
        raise KeyError(
            f"Contact sensor '{sensor_name}' not found in scene.sensors. "
            f"Available sensors: {list(env.scene.sensors.keys())}"
        )

    contact_sensor = env.scene.sensors[sensor_name]
    net_contact_forces = contact_sensor.data.net_forces_w_history  # (H, N, C, 3)

    # Compute max force magnitude over history and channels.
    # Shape: (H, N, C, 3) -> (H, N, C) -> (N, C)
    force_mag = torch.norm(net_contact_forces, dim=-1)
    max_force, _ = torch.max(force_mag, dim=0)

    # Any contact above threshold counts as an undesired contact.
    is_contact = max_force > threshold  # (N, C)
    # Penalty is the number of undesired contacts per environment.
    return torch.sum(is_contact.to(torch.float32), dim=1)


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
Base height rewards.
"""


def base_height_target(
    env: "ManagerBasedRlEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward maintaining base height at target using L2 squared kernel.

    Args:
        env: The environment instance.
        target_height: Target base height in meters.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the reward (negative penalty).
    """
    entity = env.entities[asset_cfg.entity_name]
    base_pos = entity.data.root_pos_w
    
    # Compute height error
    height_error = base_pos[:, 2] - target_height
    return -torch.square(height_error)


"""
Joint rewards.
"""


def dof_similar_to_default(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward keeping joint positions similar to default positions using L2 squared kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs,) containing the reward (negative penalty).
    """
    entity = env.entities[asset_cfg.entity_name]
    joint_pos = entity.data.joint_pos
    default_joint_pos = entity.data.default_joint_pos
    
    # Compute difference from default
    joint_diff = joint_pos - default_joint_pos
    return -torch.sum(torch.square(joint_diff), dim=1)


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
    # Require a contact sensor to be present.
    if not hasattr(env.scene, "sensors"):
        raise AttributeError(
            "Scene does not have 'sensors' attribute. "
            "Contact sensor must be configured in the scene."
        )
    
    if isinstance(sensor_cfg, str):
        sensor_name = sensor_cfg
    else:
        sensor_name = getattr(sensor_cfg, "entity_name", None) or getattr(sensor_cfg, "name", None) or "contact_forces"
    
    if sensor_name not in env.scene.sensors:
        raise KeyError(
            f"Contact sensor '{sensor_name}' not found in scene.sensors. "
            f"Available sensors: {list(env.scene.sensors.keys())}"
        )

    contact_sensor = env.scene.sensors[sensor_name]

    # First-contact indicator and last air-time buffers.
    # Shapes: (N, C)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    last_air_time = contact_sensor.data.last_air_time

    # Reward long air-times that just ended in first contact.
    # (N, C) -> (N,)
    air_time_excess = (last_air_time - threshold).clamp_min(0.0)
    reward_per_link = air_time_excess * first_contact.to(air_time_excess.dtype)
    reward = torch.sum(reward_per_link, dim=1)

    # Only reward stepping behaviour when commanded velocity is non-trivial.
    cmd = env.command_manager.get_command(command_name)
    moving_mask = torch.norm(cmd[:, :2], dim=1) > 0.1
    reward = reward * moving_mask.to(reward.dtype)

    return reward
