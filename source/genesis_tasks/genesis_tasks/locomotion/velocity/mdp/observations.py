"""Common observation functions for velocity tracking locomotion tasks.

These functions can be used to define observation terms in the MDP configuration.
They follow the same interface as IsaacLab's observation functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesislab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


"""
Root state observations.
"""


def base_lin_vel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base linear velocity in world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs, 3) containing linear velocity [x, y, z].
    """
    return env.entities[asset_cfg.entity_name].data.root_lin_vel_w


def base_ang_vel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base angular velocity in world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs, 3) containing angular velocity [x, y, z].
    """
    return env.entities[asset_cfg.entity_name].data.root_ang_vel_w


def projected_gravity(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs, 3) containing gravity vector in body frame.
    """
    return env.entities[asset_cfg.entity_name].data.projected_gravity_b


"""
Joint state observations.
"""


def joint_pos(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions of the asset.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs, num_joints) containing joint positions.
    """
    entity = env.entities[asset_cfg.entity_name]
    joint_pos = entity.data.joint_pos
    
    # Filter by joint_ids if specified
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        return joint_pos[:, asset_cfg.joint_ids]
    return joint_pos


def joint_pos_rel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default joint positions.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs, num_joints) containing joint position offsets.
    """
    entity = env.entities[asset_cfg.entity_name]
    joint_pos = entity.data.joint_pos
    
    # Get default joint positions (if available, otherwise use zeros)
    if hasattr(entity.data, "default_joint_pos"):
        default_joint_pos = entity.data.default_joint_pos
    else:
        # Fallback: use current joint positions as default (results in zeros)
        default_joint_pos = joint_pos.clone()
    
    # Filter by joint_ids if specified
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        return joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]
    return joint_pos - default_joint_pos


def joint_vel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities of the asset.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs, num_joints) containing joint velocities.
    """
    entity = env.entities[asset_cfg.entity_name]
    joint_vel = entity.data.joint_vel
    
    # Filter by joint_ids if specified
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        return joint_vel[:, asset_cfg.joint_ids]
    return joint_vel


def joint_vel_rel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities relative to default joint velocities.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Tensor of shape (num_envs, num_joints) containing joint velocity offsets.
    """
    entity = env.entities[asset_cfg.entity_name]
    joint_vel = entity.data.joint_vel
    
    # Get default joint velocities (if available, otherwise use zeros)
    if hasattr(entity.data, "default_joint_vel"):
        default_joint_vel = entity.data.default_joint_vel
    else:
        # Fallback: use zeros
        default_joint_vel = torch.zeros_like(joint_vel)
    
    # Filter by joint_ids if specified
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        return joint_vel[:, asset_cfg.joint_ids] - default_joint_vel[:, asset_cfg.joint_ids]
    return joint_vel - default_joint_vel


"""
Action observations.
"""


def last_action(env: "ManagerBasedRlEnv", action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    Args:
        env: The environment instance.
        action_name: The name of the action term. If None, returns the entire action tensor.

    Returns:
        Tensor of shape (num_envs, action_dim) containing the last actions.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_action


"""
Command observations.
"""


def generated_commands(env: "ManagerBasedRlEnv", command_name: str | None = None) -> torch.Tensor:
    """The generated command from command term in the command manager.

    Args:
        env: The environment instance.
        command_name: The name of the command term. If None, returns the first available command.

    Returns:
        Tensor of shape (num_envs, command_dim) containing the current commands.
    """
    if command_name is None:
        # Try to get the first available command
        if hasattr(env, "command_manager") and hasattr(env.command_manager, "_terms"):
            if len(env.command_manager._terms) > 0:
                command_name = list(env.command_manager._terms.keys())[0]
            else:
                # Fallback: return zeros
                return torch.zeros((env.num_envs, 1), device=env.device)
        else:
            return torch.zeros((env.num_envs, 1), device=env.device)
    
    # During initialization, commands may not be ready yet
    # Return zeros as a placeholder
    if not hasattr(env, 'command_manager') or command_name not in getattr(env.command_manager, '_terms', {}):
        num_envs = env.num_envs
        # Return a default command shape (assuming 3D velocity command: lin_vel_x, lin_vel_y, ang_vel_z)
        return torch.zeros((num_envs, 3), device=env.device)
    return env.command_manager.get_command(command_name)


"""
Sensor observations.
"""


def height_scan(env: "ManagerBasedRlEnv", sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the sensor entity.
        offset: Offset to subtract from the returned values. Defaults to 0.5.

    Returns:
        Tensor of shape (num_envs, num_rays) containing height scan values.
    """
    # TODO: Implement height scan when RayCaster sensor is available
    # For now, return zeros as placeholder
    # sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    
    # Placeholder: return zeros
    return torch.zeros((env.num_envs, 1), device=env.device)
