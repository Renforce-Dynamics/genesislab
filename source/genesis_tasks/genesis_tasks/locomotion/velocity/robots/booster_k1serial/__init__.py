"""Booster K1Serial velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import BoosterK1SerialFlatEnvCfg, BoosterK1SerialFlatEnvCfg_PLAY
from .rough_env_cfg import BoosterK1SerialRoughEnvCfg, BoosterK1SerialRoughEnvCfg_PLAY

__all__ = [
    "BoosterK1SerialFlatEnvCfg",
    "BoosterK1SerialFlatEnvCfg_PLAY",
    "BoosterK1SerialRoughEnvCfg",
    "BoosterK1SerialRoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Booster-K1Serial-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:BoosterK1SerialFlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Booster-K1Serial-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:BoosterK1SerialFlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Booster-K1Serial-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:BoosterK1SerialRoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Booster-K1Serial-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:BoosterK1SerialRoughEnvCfg_PLAY",
    },
)
