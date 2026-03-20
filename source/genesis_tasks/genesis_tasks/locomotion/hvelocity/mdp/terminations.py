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
    fallen = base_pos[:, 2] < threshold
    return fallen


def root_height_below_minimum(
    env: "ManagerBasedRlEnv",
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root height (world z) is below ``minimum_height`` (IsaacLab-style name)."""

    return base_height(env, asset_cfg=asset_cfg, threshold=minimum_height)


def illegal_contact(env: "ManagerBasedRlEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when illegal contacts are detected above a threshold.

    Args:
        env: The environment instance.
        threshold: Force threshold for contact detection.
        sensor_cfg: Configuration for the contact sensor. Should have body_ids
            or body_names set to filter which bodies to check.

    Returns:
        Boolean tensor of shape (num_envs,) indicating terminated environments.
    """
    # Get sensor name (support both entity_name and name for compatibility)
    sensor_name = getattr(sensor_cfg, "name", None) or getattr(sensor_cfg, "entity_name", None)
    if sensor_name is None:
        sensor_name = "contact_forces"

    if sensor_name not in env.scene.sensors:
        raise KeyError(
            f"Contact sensor '{sensor_name}' not found in scene.sensors. "
            f"Available sensors: {list(env.scene.sensors.keys())}"
        )

    contact_sensor = env.scene.sensors[sensor_name]
    net_contact_forces = contact_sensor.data.net_forces_w_history  # (H, N, C, 3)

    # Filter by body_ids if specified
    body_ids = getattr(sensor_cfg, "body_ids", slice(None))
    if body_ids != slice(None):
        # Index the channel dimension (third dimension)
        net_contact_forces = net_contact_forces[:, :, body_ids, :]  # (H, N, len(body_ids), 3)

    # Compute max force magnitude over history and channels.
    force_mag = torch.norm(net_contact_forces, dim=-1)  # (H, N, C) or (H, N, len(body_ids))
    max_force, _ = torch.max(force_mag, dim=0)  # (N, C) or (N, len(body_ids))

    # Terminate if any channel exceeds the threshold.
    is_contact = max_force > threshold  # (N, C) or (N, len(body_ids))
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


def bad_orientation_radians(
    env: "ManagerBasedRlEnv",
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base tilt exceeds ``limit_angle`` (radians), from projected gravity."""

    entity = env.entities[asset_cfg.entity_name]
    g = entity.data.projected_gravity_b
    tilt = torch.atan2(torch.norm(g[:, :2], dim=1), torch.abs(g[:, 2].clamp(min=1e-6)))
    return tilt > limit_angle


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
    if not hasattr(env.gsscene, "cfg"):
        raise AttributeError(
            "Scene does not have 'cfg' attribute. "
            "Cannot determine terrain type for bounds checking."
        )
    
    if not hasattr(env.gsscene.cfg, "terrain"):
        raise AttributeError(
            "Scene config does not have 'terrain' attribute. "
            "Cannot determine terrain type for bounds checking."
        )
    
    terrain_cfg = env.gsscene.cfg.terrain
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
