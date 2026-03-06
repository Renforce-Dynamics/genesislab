"""Configuration for Unitree G1 humanoid robot.

Reference: https://github.com/unitreerobotics/unitree_ros

This configuration includes PD parameters and actuator settings copied from robotlib.
The actuator configuration follows the structure from robotlib/robotlib/beyondMimic/robots/g1.py.
"""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Try to import robotlib asset path, but allow it to be None if not available
try:
    from robotlib import ROBOTLIB_ASSETLIB_DIR as ASSET_DIR
except ImportError:
    ASSET_DIR = None

##
# PD Parameters (copied from robotlib)
##
# These values are used for actuator configuration in robotlib
# Natural frequency and damping ratio for PD control
# Note: robotlib uses 10Hz (10 * 2 * pi) natural frequency
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

G1_OPENSOURCE_CFG = RobotCfg(
    morph_type="URDF",
    morph_path=f"{ASSET_DIR}/unitree/unitree_g1/urdf/g1_29dof.urdf" if ASSET_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.76],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    # Joint names to control (all actuated joints by default)
    control_dofs=None,
    # PD gains per joint group (matching robotlib actuator configuration)
    # Note: In robotlib, these are configured via actuators dict with ImplicitActuatorCfg
    # Here we provide them as pd_gains for reference
    pd_gains={
        # Legs (hip_yaw, hip_roll, hip_pitch, knee)
        ".*_hip_pitch_joint": (STIFFNESS_7520_14, DAMPING_7520_14),
        ".*_hip_roll_joint": (STIFFNESS_7520_22, DAMPING_7520_22),
        ".*_hip_yaw_joint": (STIFFNESS_7520_14, DAMPING_7520_14),
        ".*_knee_joint": (STIFFNESS_7520_22, DAMPING_7520_22),
        # Feet (ankle_pitch, ankle_roll)
        ".*_ankle_pitch_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020),
        ".*_ankle_roll_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020),
        # Waist
        "waist_roll_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020),
        "waist_pitch_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020),
        "waist_yaw_joint": (STIFFNESS_7520_14, DAMPING_7520_14),
        # Arms (shoulder, elbow, wrist)
        ".*_shoulder_pitch_joint": (STIFFNESS_5020, DAMPING_5020),
        ".*_shoulder_roll_joint": (STIFFNESS_5020, DAMPING_5020),
        ".*_shoulder_yaw_joint": (STIFFNESS_5020, DAMPING_5020),
        ".*_elbow_joint": (STIFFNESS_5020, DAMPING_5020),
        ".*_wrist_roll_joint": (STIFFNESS_5020, DAMPING_5020),
        ".*_wrist_pitch_joint": (STIFFNESS_4010, DAMPING_4010),
        ".*_wrist_yaw_joint": (STIFFNESS_4010, DAMPING_4010),
    },
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={
        # Additional options for URDF loading
        "replace_cylinders_with_capsules": True,
    },
)

# Note: For full actuator configuration matching robotlib, you would need to use
# genesislab.components.actuators.ImplicitActuatorCfg with the following structure:
#
# actuators = {
#     "legs": ImplicitActuatorCfg(
#         joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
#         effort_limit_sim={...},
#         velocity_limit_sim={...},
#         stiffness={...},
#         damping={...},
#         armature={...},
#     ),
#     ...
# }
#
# This would be configured at a higher level (e.g., in ArticulationCfg) rather than in RobotCfg.

G1_CYLINDER_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ASSET_DIR}/unitree/unitree_g1/beyond/g1_29dof.usd" if ASSET_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.76],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={},
)
