"""Velocity tracking locomotion tasks for GenesisLab.

This package provides velocity tracking locomotion environments following the same
design patterns as mjlab/IsaacLab's velocity locomotion tasks.

Main exports:
    - VelocityEnvCfg: Base configuration class for velocity tracking tasks
    - CommandsCfg, ActionsCfg, ObservationsCfg, RewardsCfg, TerminationsCfg, CurriculumCfg:
      MDP component configurations
    - mdp: MDP functions module (observations, rewards, terminations, curriculums, commands, actions)
"""

from .velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    VelocityEnvCfg,
)

__all__ = [
    "VelocityEnvCfg",
    "CommandsCfg",
    "ActionsCfg",
    "ObservationsCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "CurriculumCfg",
]
