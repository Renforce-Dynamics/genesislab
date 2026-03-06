"""Booster K1 velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import BoosterK1FlatEnvCfg, BoosterK1FlatEnvCfg_PLAY
from .rough_env_cfg import BoosterK1RoughEnvCfg, BoosterK1RoughEnvCfg_PLAY

__all__ = [
    "BoosterK1FlatEnvCfg",
    "BoosterK1FlatEnvCfg_PLAY",
    "BoosterK1RoughEnvCfg",
    "BoosterK1RoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Booster-K1-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:BoosterK1FlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Booster-K1-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:BoosterK1FlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Booster-K1-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:BoosterK1RoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Booster-K1-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:BoosterK1RoughEnvCfg_PLAY",
    },
)
