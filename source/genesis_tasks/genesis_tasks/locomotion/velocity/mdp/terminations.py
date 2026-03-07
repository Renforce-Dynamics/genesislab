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

    Terminate the episode when the episode length exceeds the maximum episode length.

    Args:
        env: The environment instance.

    Returns:
        Boolean tensor of shape (num_envs,) indicating timed-out environments.
    """
    if env.max_episode_length is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.episode_length_buf >= env.max_episode_length


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
    force_mag = torch.norm(net_contact_forces, dim=-1)  # (H, N, C)
    max_force, _ = torch.max(force_mag, dim=0)  # (N, C)

    # Terminate if any channel exceeds the threshold.
    is_contact = max_force > threshold  # (N, C)
    terminated = torch.any(is_contact, dim=1)  # (N,)
    return terminated


def bad_orientation(
    env: "ManagerBasedRlEnv",
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base orientation exceeds limit angle.

    This checks if the robot's pitch or roll angles exceed the limit.

    Args:
        env: The environment instance.
        limit_angle: Maximum allowed pitch/roll angle in degrees.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        Boolean tensor of shape (num_envs,) indicating terminated environments.
    """
    entity = env.entities[asset_cfg.entity_name]
    projected_gravity = entity.data.projected_gravity_b
    
    # Compute pitch and roll from projected gravity
    # projected_gravity is the gravity vector in body frame
    # For a flat orientation, projected_gravity should be [0, 0, -1]
    # Pitch and roll can be computed from the xy components
    pitch_roll_mag = torch.norm(projected_gravity[:, :2], dim=1)
    
    # Convert to angle (in radians, then to degrees)
    # For small angles: angle ≈ sin(angle) ≈ |projected_gravity_xy|
    # For larger angles, we use atan2
    angle_rad = torch.atan2(pitch_roll_mag, torch.abs(projected_gravity[:, 2]))
    angle_deg = torch.rad2deg(angle_rad)
    
    # Terminate if angle exceeds limit
    terminated = angle_deg > limit_angle
    return terminated


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
    if not hasattr(env.scene, "cfg"):
        raise AttributeError(
            "Scene does not have 'cfg' attribute. "
            "Cannot determine terrain type for bounds checking."
        )
    
    if not hasattr(env.scene.cfg, "terrain"):
        raise AttributeError(
            "Scene config does not have 'terrain' attribute. "
            "Cannot determine terrain type for bounds checking."
        )
    
    terrain_cfg = env.scene.cfg.terrain
    if terrain_cfg is None:
        # No terrain configured, no bounds to check
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    if hasattr(terrain_cfg, "type"):
        if terrain_cfg.type == "plane":
            # Infinite terrain (plane), no bounds
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        elif terrain_cfg.type == "generator":
            # Terrain generator - bounds checking not yet implemented
            raise NotImplementedError(
                "Terrain bounds checking is not yet implemented for generator terrain. "
                "This termination term requires terrain bounds API from the terrain generator."
            )
        else:
            raise ValueError(
                f"Unknown terrain type '{terrain_cfg.type}'. "
                f"Expected 'plane' or 'generator'."
            )
    
    # If terrain_cfg exists but has no type, assume plane (infinite)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
