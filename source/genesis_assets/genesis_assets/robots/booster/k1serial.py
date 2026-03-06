"""Configuration for Booster K1 Serial humanoid robot (USD variant).

Reference: Booster humanoid robots from BeyondMimic.
"""

from __future__ import annotations

from genesislab.components.actuators import IdealPDActuatorCfg
from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_USD_DIR

##
# Configuration
##

BOOSTER_K1SERIAL_22DOF_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{GENESIS_ASSETS_USD_DIR}/booster_k1_rev/usd/K1_serial.usd",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.53],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    actuators={
        "Head": IdealPDActuatorCfg(
            joint_names_expr=[".*Head_Yaw.*", ".*Head_Pitch.*"],
            effort_limit_sim=7.0,
            stiffness=20.0,
            damping=0.2,
            armature=0.01,
        ),
        "Arms": IdealPDActuatorCfg(
            joint_names_expr=[".*Shoulder_Pitch.*", ".*Shoulder_Roll.*", ".*Shoulder_Yaw.*", ".*Elbow.*"],
            effort_limit_sim=10.0,
            stiffness=20.0,
            damping=0.2,
            armature=0.01,
        ),
        "Legs": IdealPDActuatorCfg(
            joint_names_expr=[".*Hip_Pitch.*", ".*Hip_Roll.*", ".*Hip_Yaw.*", ".*Knee.*", ".*Ankle_Up.*", ".*Ankle_Down.*"],
            effort_limit_sim={
                ".*Hip_Pitch.*": 60.0,
                ".*Hip_Roll.*": 25.0,
                ".*Hip_Yaw.*": 30.0,
                ".*Knee.*": 60.0,
                ".*Ankle_Up.*": 24.0,
                ".*Ankle_Down.*": 15.0,
            },
            stiffness={
                ".*Hip_Pitch.*": 100.0,
                ".*Hip_Roll.*": 100.0,
                ".*Hip_Yaw.*": 100.0,
                ".*Knee.*": 100.0,
                ".*Ankle_Up.*": 50.0,
                ".*Ankle_Down.*": 50.0,
            },
            damping={
                ".*Hip_Pitch.*": 5.0,
                ".*Hip_Roll.*": 5.0,
                ".*Hip_Yaw.*": 5.0,
                ".*Knee.*": 5.0,
                ".*Ankle_Up.*": 3.0,
                ".*Ankle_Down.*": 3.0,
            },
            armature=0.01,
        ),
    },
    morph_options={},
)
