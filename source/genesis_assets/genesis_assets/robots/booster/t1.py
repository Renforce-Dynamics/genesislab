"""Configuration for Booster T1 humanoid robot.

Reference: Booster humanoid robots from BeyondMimic.
"""

from __future__ import annotations

from genesislab.components.actuators import ImplicitActuatorCfg
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

##
# Configuration
##

BOOSTER_T1_CFG = RobotCfg(
    morph_type="URDF",
    morph_path=f"{ASSET_DIR}/robots/T1/T1_23dof.urdf",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.70],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim={
                ".*_Shoulder_Pitch": 18.0,
                ".*_Shoulder_Roll": 18.0,
                ".*_Elbow_Pitch": 18.0,
                ".*_Elbow_Yaw": 18.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_Pitch": 7.33,
                ".*_Shoulder_Roll": 7.33,
                ".*_Elbow_Pitch": 7.33,
                ".*_Elbow_Yaw": 7.33,
            },
            stiffness={
                ".*_Shoulder_Pitch": 50.0,
                ".*_Shoulder_Roll": 50.0,
                ".*_Elbow_Pitch": 50.0,
                ".*_Elbow_Yaw": 50.0,
            },
            damping={
                ".*_Shoulder_Pitch": 1.0,
                ".*_Shoulder_Roll": 1.0,
                ".*_Elbow_Pitch": 1.0,
                ".*_Elbow_Yaw": 1.0,
            },
            armature={
                ".*_Shoulder_Pitch": ARMATURE_4310,
                ".*_Shoulder_Roll": ARMATURE_4310,
                ".*_Elbow_Pitch": ARMATURE_4310,
                ".*_Elbow_Yaw": ARMATURE_4310,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=25.0,
            velocity_limit_sim=12.57,
            stiffness=200.0,
            damping=5.0,
            armature=ARMATURE_6408,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 25.0,
                ".*_Hip_Yaw": 25.0,
                ".*_Knee_Pitch": 60.0,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 16.76,
                ".*_Hip_Roll": 12.57,
                ".*_Hip_Yaw": 12.57,
                ".*_Knee_Pitch": 12.57,
            },
            stiffness={
                ".*_Hip_Pitch": 200.0,
                ".*_Hip_Roll": 200.0,
                ".*_Hip_Yaw": 200.0,
                ".*_Knee_Pitch": 200.0,
            },
            damping={
                ".*_Hip_Pitch": 5.0,
                ".*_Hip_Roll": 5.0,
                ".*_Hip_Yaw": 5.0,
                ".*_Knee_Pitch": 5.0,
            },
            armature={
                ".*_Hip_Pitch": ARMATURE_8112,
                ".*_Hip_Roll": ARMATURE_6408,
                ".*_Hip_Yaw": ARMATURE_6408,
                ".*_Knee_Pitch": ARMATURE_8116,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Ankle_Pitch",
                ".*_Ankle_Roll"
            ],
            effort_limit_sim={
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            stiffness=50.0,
            damping=1.0,
            armature=2 * ARMATURE_4315,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[".*Head.*"],
            effort_limit_sim=7.0,
            velocity_limit_sim=20.0,
            stiffness=10.0,
            damping=1.0,
            armature=0.001,
        ),
    },
    morph_options={
        "replace_cylinders_with_capsules": False,
    },
)
