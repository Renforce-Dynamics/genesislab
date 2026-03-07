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

from genesislab.components.sensors import ContactSensorCfg

##
# MDP settings (configclass-based, for direct use in task configs)
##

@configclass
class VelocitySceneCfg(SceneCfg):
    num_envs    : int = 4096
    env_spacing : tuple = (2.5, 2.5)
    dt          : float = 0.005
    substeps    : int = 1
    backend     : str = "cuda"
    viewer      : bool = False

    terrain     : TerrainCfg =TerrainCfg(type="rough")
    robots      : dict = {"robot": None}
    sensors: dict = {
        "contact_forces": ContactSensorCfg(
            entity_name="robot",
            history_length=3,
            track_air_time=True,
        )
    }

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity: mdp.UniformVelocityCommandCfg = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        init_velocity_prob=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )
    """Base velocity command configuration."""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )
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

    # Contact / gait-related terms (kept for IsaacLab compatibility; currently no contact sensors).
    # NOTE: Our SceneEntityCfg is Genesis-specific and currently only carries `entity_name`.
    # We therefore keep contact/body selection logic inside the reward functions themselves.
    feet_air_time: RewardTermCfg | None = RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": "contact_forces",
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts: RewardTermCfg | None = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": "contact_forces",
            "threshold": 1.0,
        },
    )

    # Optional penalties
    flat_orientation_l2: RewardTermCfg = RewardTermCfg(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits: RewardTermCfg = RewardTermCfg(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # base_height: TerminationTermCfg = TerminationTermCfg(
    #     func=mdp.base_height,
    #     time_out=False,
    #     params={"threshold": 0.15, "asset_cfg": SceneEntityCfg("robot")},
    # )

    # IsaacLab-style contact-based termination (currently a no-op without contact sensors,
    # but kept for configuration compatibility).
    base_contact: TerminationTermCfg | None = TerminationTermCfg(
        func=mdp.illegal_contact,
        time_out=False,
        params={"sensor_cfg": "contact_forces", "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels: CurriculumTermCfg | None = None
    """Terrain levels curriculum (optional)."""
