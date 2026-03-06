"""Smplx velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import SMPLXFlatEnvCfg, SMPLXFlatEnvCfg_PLAY
from .rough_env_cfg import SMPLXRoughEnvCfg, SMPLXRoughEnvCfg_PLAY

__all__ = [
    "SMPLXFlatEnvCfg",
    "SMPLXFlatEnvCfg_PLAY",
    "SMPLXRoughEnvCfg",
    "SMPLXRoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Smplx-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SMPLXFlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Smplx-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SMPLXFlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Smplx-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SMPLXRoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Smplx-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SMPLXRoughEnvCfg_PLAY",
    },
)
