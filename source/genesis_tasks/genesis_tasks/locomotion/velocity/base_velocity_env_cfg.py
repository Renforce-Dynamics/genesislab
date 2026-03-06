"""Base configuration for velocity tracking locomotion tasks.

This module provides a base configuration class that can be inherited by robot-specific
configurations, following IsaacLab's design pattern.
"""

import math
from dataclasses import MISSING

from genesislab.components.entities.scene_cfg import SceneCfg, TerrainCfg
from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genesislab.managers import SceneEntityCfg
from genesislab.utils.configclass import configclass

from .velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
import genesis_tasks.locomotion.velocity.mdp as mdp


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRlEnvCfg):
    """Base configuration for velocity tracking locomotion tasks on rough terrain.

    This class provides a complete structure for velocity tracking locomotion tasks.
    Robot-specific configurations should inherit from this class and override the
    necessary fields in their __post_init__ method.
    """

    # Scene settings
    scene: SceneCfg = MISSING
    """Scene configuration including robots, terrain, and sensors."""

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    """Observation specifications."""

    actions: ActionsCfg = MISSING
    """Action specifications."""

    commands: CommandsCfg = MISSING
    """Command specifications."""

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    """Reward terms."""

    terminations: TerminationsCfg = TerminationsCfg()
    """Termination terms."""

    curriculum: CurriculumCfg | None = CurriculumCfg()
    """Curriculum terms."""

    def __post_init__(self):
        """Post initialization."""
        # General settings
        if not hasattr(self, "decimation") or self.decimation is None:
            self.decimation = 4
        if not hasattr(self, "episode_length_s") or self.episode_length_s is None:
            self.episode_length_s = 20.0

        # Initialize default commands if not set
        if isinstance(self.commands, type(MISSING)) or self.commands is MISSING:
            self.commands = CommandsCfg(
                base_velocity=mdp.UniformVelocityCommandCfg(
                    asset_name="robot",
                    resampling_time_range=(10.0, 10.0),
                    rel_standing_envs=0.02,
                    rel_heading_envs=1.0,
                    heading_command=True,
                    heading_control_stiffness=0.5,
                    debug_vis=True,
                    ranges=mdp.UniformVelocityCommandCfg.Ranges(
                        lin_vel_x=(-1.0, 1.0),
                        lin_vel_y=(-1.0, 1.0),
                        ang_vel_z=(-1.0, 1.0),
                        heading=(-math.pi, math.pi),
                    ),
                )
            )

        # Initialize default actions if not set
        if isinstance(self.actions, type(MISSING)) or self.actions is MISSING:
            self.actions = ActionsCfg(
                joint_pos=mdp.JointPositionActionCfg(
                    asset_name="robot",
                    joint_names=[".*"],
                    scale=0.5,
                    use_default_offset=True,
                )
            )

        # Check if terrain levels curriculum is enabled
        scene_is_missing = isinstance(self.scene, type(MISSING)) or self.scene is MISSING
        if not scene_is_missing and self.curriculum is not None and hasattr(self.curriculum, "terrain_levels"):
            # Enable curriculum for terrain generator if available
            if hasattr(self.scene, "terrain") and self.scene.terrain is not None:
                if hasattr(self.scene.terrain, "terrain_generator"):
                    terrain_gen = self.scene.terrain.terrain_generator
                    if terrain_gen is not None and hasattr(terrain_gen, "curriculum"):
                        terrain_gen.curriculum = True
