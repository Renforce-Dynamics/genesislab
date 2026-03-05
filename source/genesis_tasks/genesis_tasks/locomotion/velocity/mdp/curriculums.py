"""Common curriculum functions for velocity tracking locomotion tasks.

These functions can be used to create curriculum for the learning environment.
They follow the same interface as IsaacLab's curriculum functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from genesislab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


def terrain_levels_vel(
    env: "ManagerBasedRlEnv", env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough
    and decrease the difficulty when the robot walks less than half of the distance required by
    the commanded velocity.

    Args:
        env: The environment instance.
        env_ids: Environment indices to update.
        asset_cfg: Configuration for the asset entity. Defaults to "robot".

    Returns:
        The mean terrain level for the given environment ids.
    """
    # TODO: Implement when terrain generator and curriculum are available
    # For now, return zeros as placeholder
    # entity = env.entities[asset_cfg.name]
    # terrain: TerrainImporter = env.scene.terrain
    # command = env.command_manager.get_command("base_velocity")
    # distance = torch.norm(entity.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    # move_down *= ~move_up
    # terrain.update_env_origins(env_ids, move_up, move_down)
    # return torch.mean(terrain.terrain_levels.float())
    return torch.tensor(0.0, device=env.device)
