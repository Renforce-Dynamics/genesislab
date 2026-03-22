"""Configuration for Unitree G1 humanoid robot (BeyondMimic URDF variant).

Reference: https://github.com/unitreerobotics/unitree_ros

This configuration includes PD parameters and actuator settings.
The actuator configuration follows the structure from robotlib/beyondMimic/robots/g1.py.
"""

from __future__ import annotations

from genesislab.components.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from genesislab.engine.assets.robot import InitialPoseCfg, RobotCfg

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

G1_FULL_ACT_CFG = RobotCfg(
    morph_type="MJCF",
    morph_path=f"{ASSET_DIR}/unitree/unitree_g1/mjcf/g1_29dof_rev_1_0.xml",
    initial_pose=InitialPoseCfg(
        pos=[0.0, 0.0, 0.36],
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

