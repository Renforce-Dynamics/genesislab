"""Configuration for Unitree G1 humanoid robot (BeyondMimic URDF variant).

Reference: https://github.com/unitreerobotics/unitree_ros

This configuration includes PD parameters and actuator settings.
The actuator configuration follows the structure from robotlib/beyondMimic/robots/g1.py.
"""

from __future__ import annotations

from genesislab.components.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from genesislab.engine.assets.robot_cfg import InitialPoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_ASSETLIB_DIR as ASSET_DIR

##
# PD Parameters
##
# These values are used for actuator configuration
# Natural frequency and damping ratio for PD control
# Note: Uses 10Hz (10 * 2 * pi) natural frequency
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# Armature values for different actuator types (copied from robotlib)
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# Stiffness values (computed from armature and natural frequency)
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

# Damping values (computed from armature, natural frequency, and damping ratio)
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

##
# Configuration
##

G1_BEYONDMIMIC_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ASSET_DIR}/third_party/beyondMimic/usd/g1.usd",
    initial_pose=InitialPoseCfg(
        pos=[0.0, 0.0, 0.76],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)


# --------------------------------------------------------------------------- #
# Single-actuator variant for joint-position actions
# --------------------------------------------------------------------------- #
#
# Some environments (e.g., imitation tracking) use JointPositionAction which
# expects a single actuator name controlling all joints. For G1 BeyondMimic we
# provide a convenience configuration that merges all actuator groups into a
# single ImplicitActuator.
#

G1_FULL_ACT_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ASSET_DIR}/third_party/beyondMimic/usd/g1.usd",
    initial_pose=InitialPoseCfg(
        pos=[0.0, 0.0, 0.76],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    actuators={
        "full": ImplicitActuatorCfg(
            # Union of all joint patterns from legs, feet, waist, waist_yaw, arms.
            joint_names_expr=[
                # Legs
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                # Feet
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                # Waist
                "waist_roll_joint",
                "waist_pitch_joint",
                "waist_yaw_joint",
                # Arms
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            # Effort limits: merge all groups.
            effort_limit_sim={
                # Legs
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                # Feet
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
                # Waist
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
                "waist_yaw_joint": 88.0,
                # Arms
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            # Velocity limits: merge all groups.
            velocity_limit_sim={
                # Legs
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                # Feet
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
                # Waist
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
                "waist_yaw_joint": 32.0,
                # Arms
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            # Stiffness: reuse group-specific values.
            stiffness={
                # Legs
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
                # Feet
                ".*_ankle_pitch_joint": 2.0 * STIFFNESS_5020,
                ".*_ankle_roll_joint": 2.0 * STIFFNESS_5020,
                # Waist
                "waist_roll_joint": 2.0 * STIFFNESS_5020,
                "waist_pitch_joint": 2.0 * STIFFNESS_5020,
                "waist_yaw_joint": STIFFNESS_7520_14,
                # Arms
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            # Damping: reuse group-specific values.
            damping={
                # Legs
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
                # Feet
                ".*_ankle_pitch_joint": 2.0 * DAMPING_5020,
                ".*_ankle_roll_joint": 2.0 * DAMPING_5020,
                # Waist
                "waist_roll_joint": 2.0 * DAMPING_5020,
                "waist_pitch_joint": 2.0 * DAMPING_5020,
                "waist_yaw_joint": DAMPING_7520_14,
                # Arms
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            # Armature: reuse group-specific values.
            armature={
                # Legs
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
                # Feet
                ".*_ankle_pitch_joint": 2.0 * ARMATURE_5020,
                ".*_ankle_roll_joint": 2.0 * ARMATURE_5020,
                # Waist
                "waist_roll_joint": 2.0 * ARMATURE_5020,
                "waist_pitch_joint": 2.0 * ARMATURE_5020,
                "waist_yaw_joint": ARMATURE_7520_14,
                # Arms
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)

