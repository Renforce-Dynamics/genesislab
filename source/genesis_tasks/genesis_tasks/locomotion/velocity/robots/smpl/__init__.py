"""Smpl velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import SMPLFlatEnvCfg, SMPLFlatEnvCfg_PLAY
from .rough_env_cfg import SMPLRoughEnvCfg, SMPLRoughEnvCfg_PLAY

__all__ = [
    "SMPLFlatEnvCfg",
    "SMPLFlatEnvCfg_PLAY",
    "SMPLRoughEnvCfg",
    "SMPLRoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Smpl-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SMPLFlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Smpl-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SMPLFlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Smpl-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SMPLRoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Smpl-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SMPLRoughEnvCfg_PLAY",
    },
)
