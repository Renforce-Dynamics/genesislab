"""Go2 velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import UnitreeGo2FlatEnvCfg, UnitreeGo2FlatEnvCfg_PLAY
from .rough_env_cfg import UnitreeGo2RoughEnvCfg, UnitreeGo2RoughEnvCfg_PLAY

__all__ = [
    "UnitreeGo2FlatEnvCfg",
    "UnitreeGo2FlatEnvCfg_PLAY",
    "UnitreeGo2RoughEnvCfg",
    "UnitreeGo2RoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Go2-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Go2-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Go2-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Go2-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg_PLAY",
    },
)
