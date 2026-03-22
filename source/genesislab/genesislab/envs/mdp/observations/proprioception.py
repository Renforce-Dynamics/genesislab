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
    
    # Get default joint positions - required for this observation
    if not hasattr(entity.data, "default_joint_pos"):
        raise AttributeError(
            f"Entity '{asset_cfg.entity_name}' data does not have 'default_joint_pos' attribute. "
            f"This observation term requires default joint positions from the entity."
        )
    
    default_joint_pos = entity.data.default_joint_pos
    
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
    
    # Get default joint velocities - required for this observation
    if not hasattr(entity.data, "default_joint_vel"):
        raise AttributeError(
            f"Entity '{asset_cfg.entity_name}' data does not have 'default_joint_vel' attribute. "
            f"This observation term requires default joint velocities from the entity."
        )
    
    default_joint_vel = entity.data.default_joint_vel
    
    # Filter by joint_ids if specified
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        return joint_vel[:, asset_cfg.joint_ids] - default_joint_vel[:, asset_cfg.joint_ids]
    return joint_vel - default_joint_vel


"""
Action observations.
"""


def last_action(env: "ManagerBasedRlEnv", action_name: str = None) -> torch.Tensor:
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


def generated_commands(env: "ManagerBasedRlEnv", command_name: str = None) -> torch.Tensor:
    """The generated command from command term in the command manager.

    Args:
        env: The environment instance.
        command_name: The name of the command term. If None, returns the first available command.

    Returns:
        Tensor of shape (num_envs, command_dim) containing the current commands.
    """
    if not hasattr(env, "command_manager"):
        raise AttributeError(
            "Environment does not have 'command_manager' attribute. "
            "This observation term requires a command manager to be configured."
        )
    
    if command_name is None:
        # Try to get the first available command
        if not hasattr(env.command_manager, "_terms"):
            raise AttributeError(
                "CommandManager does not have '_terms' attribute. "
                "Command manager may not be properly initialized."
            )
        
        if len(env.command_manager._terms) == 0:
            raise ValueError(
                "CommandManager has no command terms configured. "
                "At least one command term must be configured for this observation."
            )
        
        command_name = list(env.command_manager._terms.keys())[0]
    
    # Validate command exists
    if command_name not in env.command_manager._terms:
        raise ValueError(
            f"Command '{command_name}' not found in command manager. "
            f"Available commands: {list(env.command_manager._terms.keys())}"
        )
    
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
    raise NotImplementedError(
        "height_scan observation is not yet implemented. "
        "This requires a RayCaster sensor which is not yet available in GenesisLab."
    )
