"""Go2W velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import UnitreeGo2WFlatEnvCfg, UnitreeGo2WFlatEnvCfg_PLAY
from .rough_env_cfg import UnitreeGo2WRoughEnvCfg, UnitreeGo2WRoughEnvCfg_PLAY

__all__ = [
    "UnitreeGo2WFlatEnvCfg",
    "UnitreeGo2WFlatEnvCfg_PLAY",
    "UnitreeGo2WRoughEnvCfg",
    "UnitreeGo2WRoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Unitree-Go2W-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2WFlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Unitree-Go2W-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2WFlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Unitree-Go2W-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2WRoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Unitree-Go2W-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2WRoughEnvCfg_PLAY",
    },
)
