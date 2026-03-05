"""Agent (RL) configuration for Go2 velocity tasks in GenesisLab.

This module mirrors the IsaacLab layout under:
``isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents``.
"""

from .rsl_rl_ppo_cfg import Go2FlatPPORunnerCfg, Go2RoughPPORunnerCfg

__all__ = [
    "Go2FlatPPORunnerCfg",
    "Go2RoughPPORunnerCfg",
]

