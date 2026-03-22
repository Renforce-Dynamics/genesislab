"""Configuration bundles for human (biped) velocity-tracking tasks.

This module mirrors :mod:`genesis_tasks.locomotion.velocity.components` in shape
but targets humanoid morphology and SoccerLab-imported reward/event structure.
It does **not** import the quadruped ``velocity`` task implementation.
"""

from __future__ import annotations

import math

from genesislab.components.additional.noise.noise_cfg import UniformNoiseCfg
from genesislab.components.sensors.fake_sensors import FakeContactSensorCfg
from genesislab.components.terrains import GenesisTerrainMorphCfg, TerrainSurfaceCfg
from genesislab.engine.scene import SceneCfg, TerrainCfg
from genesislab.managers import EventTermCfg, SceneEntityCfg
from genesislab.managers.command_manager import CommandTermCfg
from genesislab.managers.curriculum_manager import CurriculumTermCfg
from genesislab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg
from genesislab.utils.configclass import configclass

import genesis_tasks.locomotion.hvelocity.mdp as mdp


@configclass
class HumanVelocitySceneCfg(SceneCfg):
    num_envs: int = 4096
    env_spacing: tuple = (2.5, 2.5)
    dt: float = 0.005
    substeps: int = 1
    backend: str = "cuda"
    viewer: bool = False

    terrain: TerrainCfg = TerrainCfg(
        terrain_type="genesisbase",
        terrain_details_cfg=GenesisTerrainMorphCfg(
            pos=(-12.0, -12.0, 0.0),
            n_subterrains=(1, 1),
            subterrain_size=(24.0, 24.0),
            vertical_scale=0.001,
            subterrain_types=[["random_uniform_terrain"]],
        ),
        surface_cfg=TerrainSurfaceCfg(diffuse_color=None),
    )
    robots: dict = {"robot": None}
    sensors: dict = {
        "contact_forces": FakeContactSensorCfg(
            entity_name="robot",
            history_length=3,
            track_air_time=True,
        )
    }


@configclass
class CommandsCfg:
    """Velocity commands (uniform sampling with curriculum ceilings)."""

    base_velocity: mdp.UniformLevelVelocityCommandCfg = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        init_velocity_prob=0.0,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1),
            heading=None,
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 2.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.2, 0.2),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    """Joint position commands via Genesis built-in PD (:class:`~genesislab.envs.mdp.actions.GenesisOriginalAction`).

    Targets are ``default_pose + scale * action``; no per-step torque solve in Python.
    Stiffness/damping come from the robot asset / actuator manager (``set_dofs_kp`` / ``kv``).
    """

    joint_pos: mdp.GenesisOriginalActionCfg = mdp.GenesisOriginalActionCfg(
        entity_name="robot",
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # Body-frame velocities align with ``base_velocity`` command (also in base frame).
        base_lin_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.base_lin_vel_b, clip=(-100, 100))
        base_ang_vel: ObservationTermCfg = ObservationTermCfg(
            func=mdp.base_ang_vel_b,
            scale=0.2,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            clip=(-100, 100),
        )
        projected_gravity: ObservationTermCfg = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        velocity_commands: ObservationTermCfg = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos: ObservationTermCfg = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            clip=(-100, 100),
        )
        joint_vel: ObservationTermCfg = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
            clip=(-100, 100),
        )
        actions: ObservationTermCfg = ObservationTermCfg(func=mdp.last_action, clip=(-12, 12))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    track_lin_vel_xy_exp: RewardTermCfg = RewardTermCfg(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp: RewardTermCfg = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive: RewardTermCfg = RewardTermCfg(func=mdp.is_alive, weight=0.15)

    lin_vel_z_l2: RewardTermCfg = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2: RewardTermCfg = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel: RewardTermCfg = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001)
    dof_acc_l2: RewardTermCfg = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2: RewardTermCfg = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits: RewardTermCfg = RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0)
    energy: RewardTermCfg = RewardTermCfg(func=mdp.mechanical_power_l1, weight=-2e-5)

    joint_deviation_arms: RewardTermCfg = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists: RewardTermCfg = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs: RewardTermCfg = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    flat_orientation_l2: RewardTermCfg = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.0)
    # base_height: RewardTermCfg = RewardTermCfg(
    #     func=mdp.base_height_l2,
    #     weight=-10.0,
    #     params={"target_height": 0.78},
    # )

    gait: RewardTermCfg = RewardTermCfg(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide: RewardTermCfg = RewardTermCfg(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance: RewardTermCfg = RewardTermCfg(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    undesired_contacts: RewardTermCfg = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"),
        },
    )


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_height: TerminationTermCfg = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.2},
    )
    bad_orientation: TerminationTermCfg = TerminationTermCfg(
        func=mdp.bad_orientation_radians,
        params={"limit_angle": 0.8},
    )


@configclass
class CurriculumCfg:
    terrain_levels: CurriculumTermCfg = CurriculumTermCfg(
        func=mdp.terrain_levels_vel,
        params={"command_name": "base_velocity"},
    )
    lin_vel_cmd_levels: CurriculumTermCfg = CurriculumTermCfg(
        func=mdp.lin_vel_cmd_levels,
        params={"command_name": "base_velocity", "num_steps": 8_000_000},
    )


@configclass
class EventsCfg:
    physics_material: EventTermCfg = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(entity_name="robot", body_names=".*"),
            "scale_range": (1.0, 1.0),
        },
    )

    add_base_mass: EventTermCfg = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(entity_name="robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # base_com: EventTermCfg = EventTermCfg(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg(entity_name="robot", body_names="base"),
    #         "com_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
    #     },
    # )

    base_external_force_torque: EventTermCfg = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(entity_name="robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    reset_base: EventTermCfg = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # Multiplicative scale around default pose (see ``reset_joints_by_scale``). Same idea as quadruped velocity task.
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot: EventTermCfg = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        },
    )
