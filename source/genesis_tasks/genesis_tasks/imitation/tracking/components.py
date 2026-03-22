"""Configuration components for whole-body tracking tasks on GenesisLab.

Mirrors the structure of .references/whole_body_tracking tracking_env_cfg.py
using GenesisLab scene/env/manager interfaces. Robot-specific configs (e.g. G1)
should subclass TrackingEnvCfg and set scene.robots["robot"], commands.motion.motion_file,
commands.motion.anchor_body_name, and commands.motion.body_names.
"""

from __future__ import annotations

from dataclasses import MISSING

from genesislab.engine.scene import SceneCfg, TerrainCfg
from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genesislab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg
from genesislab.managers.curriculum_manager import CurriculumTermCfg
from genesislab.managers import SceneEntityCfg, EventTermCfg
from genesislab.utils.configclass import configclass
from genesislab.components.sensors.fake_sensors import FakeContactSensorCfg
from genesislab.components.additional.noise.noise_cfg import UniformNoiseCfg

import genesis_tasks.imitation.tracking.mdp as mdp

from genesislab.engine.sim import SimOptionsCfg
from genesislab.components.terrains import GenesisTerrainMorphCfg, TerrainSurfaceCfg, FlatSubTerrainCfg

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class TrackingSceneCfg(SceneCfg):
    """Scene for whole-body tracking: flat terrain, one robot, contact sensor."""

    num_envs: int = 4096
    env_spacing: tuple[float, float] = (2.5, 2.5)
    viewer: bool = False
    sim_options: SimOptionsCfg = SimOptionsCfg(dt=0.005)
    terrain: TerrainCfg = TerrainCfg(
        terrain_type="plane",
    )
    robots: dict = {"robot": None}
    sensors: dict = {
        # "contact_forces": FakeContactSensorCfg(
        #     entity_name="robot",
        #     history_length=3,
        #     track_air_time=True,
        # )
    }


@configclass
class CommandsCfg:
    """Command specifications for tracking MDP."""

    motion: mdp.MotionCommandCfg = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file="data/datasets/dance1_subject1.npz",
        anchor_body_name=MISSING,
        body_names=MISSING,
        pose_range={
            "x": (-0.00, 0.00),
            "y": (-0.00, 0.00),
            "z": (-0.00, 0.00),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.0, 0.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for tracking MDP."""

    joint_pos: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        entity_name="robot",
        joint_names=[".*"],
        scale=1.0,
        clip=(-20, 20),
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for tracking MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy (with optional noise)."""

        command: ObservationTermCfg = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
        )
        motion_anchor_pos_b: ObservationTermCfg = ObservationTermCfg(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        )
        motion_anchor_ori_b: ObservationTermCfg = ObservationTermCfg(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel: ObservationTermCfg = ObservationTermCfg(
            func=mdp.base_lin_vel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        )
        base_ang_vel: ObservationTermCfg = ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        joint_pos: ObservationTermCfg = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        joint_vel: ObservationTermCfg = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        )
        actions: ObservationTermCfg = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObservationGroupCfg):
        """Privileged observations for critic (no noise)."""

        command: ObservationTermCfg = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
        )
        motion_anchor_pos_b: ObservationTermCfg = ObservationTermCfg(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
        )
        motion_anchor_ori_b: ObservationTermCfg = ObservationTermCfg(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
        )
        body_pos: ObservationTermCfg = ObservationTermCfg(
            func=mdp.robot_body_pos_b,
            params={"command_name": "motion"},
        )
        body_ori: ObservationTermCfg = ObservationTermCfg(
            func=mdp.robot_body_ori_b,
            params={"command_name": "motion"},
        )
        base_lin_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.base_ang_vel)
        joint_pos: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_vel_rel)
        actions: ObservationTermCfg = ObservationTermCfg(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventsCfg:
    """Event terms for tracking (startup/interval)."""

    physics_material: EventTermCfg = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(entity_name="robot", body_names=".*"),
            "scale_range": (0.3, 1.6),
        },
    )
    add_joint_default_pos: EventTermCfg = EventTermCfg(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(entity_name="robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    base_com: EventTermCfg = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(entity_name="robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    push_robot: EventTermCfg = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for tracking MDP."""

    motion_global_anchor_pos: RewardTermCfg = RewardTermCfg(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori: RewardTermCfg = RewardTermCfg(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos: RewardTermCfg = RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori: RewardTermCfg = RewardTermCfg(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel: RewardTermCfg = RewardTermCfg(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel: RewardTermCfg = RewardTermCfg(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2: RewardTermCfg = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit: RewardTermCfg = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg(entity_name="robot", joint_names=[".*"])},
    )
    # undesired_contacts: RewardTermCfg = RewardTermCfg(
    #     func=mdp.undesired_contacts,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
    #         "threshold": 1.0,
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for tracking MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    anchor_pos: TerminationTermCfg = TerminationTermCfg(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori: TerminationTermCfg = TerminationTermCfg(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos: TerminationTermCfg = TerminationTermCfg(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for tracking (empty by default)."""

    pass
