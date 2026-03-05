"""Velocity tracking locomotion tasks.

This package provides velocity tracking tasks for legged robots.
Tasks are configured using configclass instead of dict for better type safety and structure.
"""

import gymnasium as gym

from .robots.go2.go2_flat_env_cfg import Go2FlatVelocityEnvCfg
from .robots.go2.go2_rough_env_cfg import Go2RoughVelocityEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Genesis-Velocity-Flat-Go2-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robots.go2.go2_flat_env_cfg:Go2FlatVelocityEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Go2-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robots.go2.go2_rough_env_cfg:Go2RoughVelocityEnvCfg",
    },
)
