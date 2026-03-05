"""Common termination functions for velocity tracking locomotion tasks.

These functions can be used to define termination terms in the MDP configuration.
They follow the same interface as IsaacLab's termination functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesislab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


def time_out(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Timeout termination condition.

    This is handled by the environment's finite horizon logic.
    Return all False here; timeouts are handled in base_env.

    Args:
        env: The environment instance.

    Returns:
        Boolean tensor of shape (num_envs,) (all False).
    """
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def base_height(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
) -> torch.Tensor:
    """Terminate when base height falls below threshold.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".
        threshold: Minimum base height before termination (meters). Defaults to 0.15.

    Returns:
        Boolean tensor of shape (num_envs,) indicating terminated environments.
    """
    entity = env.entities[asset_cfg.entity_name]
    base_pos = entity.data.root_pos_w
    
    # Check if base is too low (fallen)
    fallen = base_pos[:, 2] < threshold
    return fallen


def illegal_contact(env: "ManagerBasedRlEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when illegal contacts are detected above a threshold.

    Args:
        env: The environment instance.
        threshold: Force threshold for contact detection.
        sensor_cfg: Configuration for the contact sensor.

    Returns:
        Boolean tensor of shape (num_envs,) indicating terminated environments.
    """
    # TODO: Implement when contact sensor is available
    # For now, return all False as placeholder
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_contact_forces = contact_sensor.data.net_forces_w_history
    # is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # return torch.any(is_contact, dim=1)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def terrain_out_of_bounds(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor moves too close to the edge of the terrain.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".
        distance_buffer: Distance buffer from terrain edge. Defaults to 3.0.

    Returns:
        Boolean tensor of shape (num_envs,) indicating terminated environments.
    """
    # Check terrain type
    if hasattr(env.scene, "cfg") and hasattr(env.scene.cfg, "terrain"):
        terrain_cfg = env.scene.cfg.terrain
        if terrain_cfg is None or (hasattr(terrain_cfg, "type") and terrain_cfg.type == "plane"):
            # Infinite terrain (plane), no bounds
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        elif hasattr(terrain_cfg, "type") and terrain_cfg.type == "generator":
            # TODO: Implement terrain bounds checking when terrain generator is available
            # For now, return all False
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Default: no termination
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
