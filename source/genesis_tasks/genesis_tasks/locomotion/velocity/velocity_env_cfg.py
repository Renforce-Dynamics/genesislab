"""Base configuration for velocity tracking locomotion tasks.

This module provides base configuration classes and factory functions for velocity tracking
locomotion tasks, following the same design patterns as mjlab/IsaacLab's velocity locomotion
environments.
"""

import math
from dataclasses import MISSING

from genesislab.components.entities.scene_cfg import SceneCfg, TerrainCfg
from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genesislab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from genesislab.managers.action_manager import ActionTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg
from genesislab.managers.command_manager import CommandTermCfg
from genesislab.managers.curriculum_manager import CurriculumTermCfg
from genesislab.managers import SceneEntityCfg
from genesislab.utils.configclass import configclass

import genesis_tasks.locomotion.velocity.mdp as mdp


##
# MDP settings (configclass-based, for direct use in task configs)
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity: mdp.UniformVelocityCommandCfg = MISSING
    """Base velocity command configuration."""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos: mdp.JointPositionActionCfg = MISSING
    """Joint position action configuration."""


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # Observation terms (order preserved)
        base_lin_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity: ObservationTermCfg = ObservationTermCfg(func=mdp.projected_gravity)
        velocity_commands: ObservationTermCfg = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_vel_rel)
        actions: ObservationTermCfg = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task rewards
    track_lin_vel_xy_exp: RewardTermCfg = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp: RewardTermCfg = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # Penalties
    lin_vel_z_l2: RewardTermCfg = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2: RewardTermCfg = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2: RewardTermCfg = RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2: RewardTermCfg = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2: RewardTermCfg = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

    # Optional penalties
    flat_orientation_l2: RewardTermCfg = RewardTermCfg(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits: RewardTermCfg = RewardTermCfg(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_height: TerminationTermCfg = TerminationTermCfg(
        func=mdp.base_height,
        time_out=False,
        params={"threshold": 0.15, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels: CurriculumTermCfg | None = None
    """Terrain levels curriculum (optional)."""


##
# Environment configuration
##


@configclass
class VelocityEnvCfg(ManagerBasedRlEnvCfg):
    """Base configuration for velocity tracking locomotion tasks.

    This base class provides a complete structure for velocity tracking locomotion tasks,
    following the same design patterns as mjlab/IsaacLab's velocity locomotion environments.

    Subclasses should override the configclass fields to customize the task for specific robots.
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

    curriculum: CurriculumCfg | None = None
    """Curriculum terms (optional)."""

    def __post_init__(self):
        """Post initialization."""
        # General settings
        if not hasattr(self, "decimation") or self.decimation is None:
            self.decimation = 4
        if not hasattr(self, "episode_length_s") or self.episode_length_s is None:
            self.episode_length_s = 20.0

        # Check if terrain levels curriculum is enabled
        scene_is_missing = isinstance(self.scene, type(MISSING)) or self.scene is MISSING
        if not scene_is_missing and self.curriculum is not None and hasattr(self.curriculum, "terrain_levels"):
            # Enable curriculum for terrain generator if available
            if hasattr(self.scene, "terrain") and self.scene.terrain is not None:
                if hasattr(self.scene.terrain, "terrain_generator"):
                    terrain_gen = self.scene.terrain.terrain_generator
                    if terrain_gen is not None and hasattr(terrain_gen, "curriculum"):
                        terrain_gen.curriculum = True
