"""Configuration for Booster K1 humanoid robot.

Reference: Booster humanoid robots from BeyondMimic.
"""

from __future__ import annotations

from genesislab.components.actuators.actuator_robotlib import DelayedImplicitActuatorCfg
from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_ASSETLIB_DIR as ASSET_DIR

##
# Actuator Parameters (from robotlib)
##
ARMATURE_6416 = 0.095625
ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.0478125
ARMATURE_4315 = 0.0339552
ARMATURE_8112 = 0.0523908
ARMATURE_8116 = 0.0636012
ARMATURE_ROB_14 = 0.001

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_6416 = 80.0
STIFFNESS_4310 = 80.0
STIFFNESS_6408 = 80.0
STIFFNESS_4315 = 80.0
STIFFNESS_ROB_14 = 4.0

DAMPING_6416 = 2.0
DAMPING_4310 = 2.0
DAMPING_6408 = 2.0
DAMPING_4315 = 2.0
DAMPING_ROB_14 = 1.0

##
# Configuration
##

BOOSTER_K1_CFG = RobotCfg(
    morph_type="URDF",
    morph_path=f"{ASSET_DIR}/robots/K1/K1_22dof.urdf",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.57],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 30.0,
                ".*_Hip_Roll": 35.0,
                ".*_Hip_Yaw": 20.0,
                ".*_Knee_Pitch": 40.0,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 8.0,
                ".*_Hip_Roll": 12.9,
                ".*_Hip_Yaw": 18.0,
                ".*_Knee_Pitch": 12.5,
            },
            stiffness={
                ".*_Hip_Pitch": STIFFNESS_6408,
                ".*_Hip_Roll": STIFFNESS_4315,
                ".*_Hip_Yaw": STIFFNESS_4310,
                ".*_Knee_Pitch": STIFFNESS_6416,
            },
            damping={
                ".*_Hip_Pitch": DAMPING_6408,
                ".*_Hip_Roll": DAMPING_4315,
                ".*_Hip_Yaw": DAMPING_4310,
                ".*_Knee_Pitch": DAMPING_6416,
            },
            armature={
                ".*_Hip_Pitch": ARMATURE_6408,
                ".*_Hip_Roll": ARMATURE_4315,
                ".*_Hip_Yaw": ARMATURE_4310,
                ".*_Knee_Pitch": ARMATURE_6416,
            },
        ),
        "feet": DelayedImplicitActuatorCfg(
            max_delay=8,
            min_delay=2,
            effort_limit_sim=20.0,
            velocity_limit_sim=18.0,
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            stiffness=30.0,
            damping=2.0,
            armature=2.0 * ARMATURE_4310,
        ),
        "arms": DelayedImplicitActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim={
                ".*_Shoulder_Pitch": 14.0,
                ".*_Shoulder_Roll": 14.0,
                ".*_Elbow_Pitch": 14.0,
                ".*_Elbow_Yaw": 14.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_Pitch": 18.0,
                ".*_Shoulder_Roll": 18.0,
                ".*_Elbow_Pitch": 18.0,
                ".*_Elbow_Yaw": 18.0,
            },
            stiffness={
                ".*_Shoulder_Pitch": STIFFNESS_ROB_14,
                ".*_Shoulder_Roll": STIFFNESS_ROB_14,
                ".*_Elbow_Pitch": STIFFNESS_ROB_14,
                ".*_Elbow_Yaw": STIFFNESS_ROB_14,
            },
            damping={
                ".*_Shoulder_Pitch": DAMPING_ROB_14,
                ".*_Shoulder_Roll": DAMPING_ROB_14,
                ".*_Elbow_Pitch": DAMPING_ROB_14,
                ".*_Elbow_Yaw": DAMPING_ROB_14,
            },
            armature={
                ".*_Shoulder_Pitch": ARMATURE_ROB_14,
                ".*_Shoulder_Roll": ARMATURE_ROB_14,
                ".*_Elbow_Pitch": ARMATURE_ROB_14,
                ".*_Elbow_Yaw": ARMATURE_ROB_14,
            },
        ),
        "head": DelayedImplicitActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*Head.*"],
            effort_limit_sim=6.0,
            velocity_limit_sim=20.0,
            stiffness=4.0,
            damping=1.0,
            armature=0.001,
        ),
    },
    morph_options={
        "replace_cylinders_with_capsules": False,
    },
)
