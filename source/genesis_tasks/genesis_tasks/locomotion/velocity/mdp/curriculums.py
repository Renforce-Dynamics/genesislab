"""Common curriculum functions for velocity tracking locomotion tasks.

These functions can be used to create curriculum for the learning environment.
They follow the same interface as mjlab/IsaacLab's curriculum functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from genesislab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv

from .commands import UniformVelocityCommandCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
    """Velocity stage configuration for curriculum."""

    step: int
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | Sequence[int],
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough
    and decrease the difficulty when the robot walks less than half of the distance required by
    the commanded velocity.

    Args:
        env: The environment instance.
        env_ids: Environment indices to update.
        command_name: Name of the velocity command term.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        The mean terrain level for the given environment ids.
    """
    # Convert env_ids to tensor if needed
    if isinstance(env_ids, Sequence) and not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)

    entity = env.entities[asset_cfg.entity_name]

    # Check if terrain generator is available
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        # No terrain curriculum available
        return torch.tensor(0.0, device=env.device)

    terrain_generator = getattr(terrain, "terrain_generator", None)
    if terrain_generator is None:
        return torch.tensor(0.0, device=env.device)

    command = env.command_manager.get_command(command_name)
    if command is None:
        return torch.tensor(0.0, device=env.device)

    # Compute the distance the robot walked
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        # Fallback: use origin
        env_origins = torch.zeros((env.num_envs, 3), device=env.device)

    distance = torch.norm(
        entity.data.root_pos_w[env_ids, :2] - env_origins[env_ids, :2], dim=1
    )

    # Robots that walked far enough progress to harder terrains
    terrain_size = getattr(terrain_generator, "size", [10.0, 10.0])
    if isinstance(terrain_size, (list, tuple)) and len(terrain_size) >= 1:
        move_up = distance > terrain_size[0] / 2
    else:
        move_up = torch.zeros_like(distance, dtype=torch.bool)

    # Robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # Update terrain levels (if method exists)
    if hasattr(terrain, "update_env_origins"):
        terrain.update_env_origins(env_ids, move_up, move_down)

    # Return mean terrain level
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is not None:
        return torch.mean(terrain_levels.float())
    return torch.tensor(0.0, device=env.device)


def commands_vel(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | Sequence[int],
    command_name: str,
    velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
    """Update command velocity ranges based on training step stages.

    Args:
        env: The environment instance.
        env_ids: Environment indices (unused, kept for API compatibility).
        command_name: Name of the velocity command term.
        velocity_stages: List of velocity stages with step thresholds and range updates.

    Returns:
        Dictionary of current velocity range values for logging.
    """
    del env_ids  # Unused

    command_term = env.command_manager.get_term(command_name)
    if command_term is None:
        # Return default values if command not found
        return {
            "lin_vel_x_min": torch.tensor(0.0),
            "lin_vel_x_max": torch.tensor(0.0),
            "lin_vel_y_min": torch.tensor(0.0),
            "lin_vel_y_max": torch.tensor(0.0),
            "ang_vel_z_min": torch.tensor(0.0),
            "ang_vel_z_max": torch.tensor(0.0),
        }

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # Apply stages based on current step counter
    for stage in velocity_stages:
        if env.common_step_counter >= stage["step"]:
            if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                cfg.ranges.lin_vel_x = stage["lin_vel_x"]
            if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                cfg.ranges.lin_vel_y = stage["lin_vel_y"]
            if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                cfg.ranges.ang_vel_z = stage["ang_vel_z"]

    # Return current ranges for logging
    return {
        "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0], device=env.device),
        "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1], device=env.device),
        "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0], device=env.device),
        "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1], device=env.device),
        "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0], device=env.device),
        "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1], device=env.device),
    }
