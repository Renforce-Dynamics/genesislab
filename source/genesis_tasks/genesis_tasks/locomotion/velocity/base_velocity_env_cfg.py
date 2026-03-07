"""Base configuration for velocity tracking locomotion tasks.

This module provides a base configuration class that can be inherited by robot-specific
configurations, following IsaacLab's design pattern.
"""

import math
from dataclasses import MISSING

from genesislab.components.entities.scene_cfg import SceneCfg, TerrainCfg
from genesislab.components.sensors import ContactSensorCfg
from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genesislab.managers import SceneEntityCfg
from genesislab.utils.configclass import configclass

from .components import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    VelocitySceneCfg
)
import genesis_tasks.locomotion.velocity.mdp as mdp


@configclass
class BaseVelocityEnvCfg(ManagerBasedRlEnvCfg):
    """Base configuration for velocity tracking locomotion tasks on rough terrain.

    This class provides a complete structure for velocity tracking locomotion tasks.
    Robot-specific configurations should inherit from this class and override the
    necessary fields in their __post_init__ method.
    """

    # Scene settings
    scene: VelocitySceneCfg = VelocitySceneCfg()
    """Scene configuration including robots, terrain, and sensors."""

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    """Observation specifications."""

    actions: ActionsCfg = ActionsCfg()
    """Action specifications."""

    commands: CommandsCfg = CommandsCfg()
    """Command specifications."""

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    """Reward terms."""

    terminations: TerminationsCfg = TerminationsCfg()
    """Termination terms."""

    curriculum: CurriculumCfg = CurriculumCfg()
    """Curriculum terms."""

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 20.0

        # Check if terrain levels curriculum is enabled
        if hasattr(self.scene, "terrain") and self.scene.terrain is not None:
            if hasattr(self.scene.terrain, "terrain_generator"):
                terrain_gen = self.scene.terrain.terrain_generator
                if terrain_gen is not None and hasattr(terrain_gen, "curriculum"):
                    terrain_gen.curriculum = True
