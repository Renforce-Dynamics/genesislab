"""Booster T1 velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import BoosterT1FlatEnvCfg, BoosterT1FlatEnvCfg_PLAY
from .rough_env_cfg import BoosterT1RoughEnvCfg, BoosterT1RoughEnvCfg_PLAY

__all__ = [
    "BoosterT1FlatEnvCfg",
    "BoosterT1FlatEnvCfg_PLAY",
    "BoosterT1RoughEnvCfg",
    "BoosterT1RoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Booster-T1-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:BoosterT1FlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Booster-T1-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:BoosterT1FlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Booster-T1-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:BoosterT1RoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Booster-T1-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:BoosterT1RoughEnvCfg_PLAY",
    },
)
