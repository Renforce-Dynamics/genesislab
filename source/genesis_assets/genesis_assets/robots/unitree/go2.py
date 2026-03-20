"""Configuration for Unitree Go2 quadruped robot.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

from __future__ import annotations

from genesislab.components.actuators.actuator_robotlib import UnitreeActuatorCfg_Go2HV
from genesislab.components.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from genesislab.engine.assets.robot_cfg import InitialPoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_UNITREE_MODEL_DIR as UNITREE_MODEL_DIR

##
# Configuration
##

UNITREE_GO2_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{UNITREE_MODEL_DIR}/unitree_go2/usd/go2.usd",
    initial_pose=InitialPoseCfg(
        pos=[0.0, 0.0, 0.4],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    default_joint_pos={
        ".*_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
    },
    actuators={
        "GO2HV": ImplicitActuatorCfg(
            joint_names_expr=[
                "FL_.*_joint",
                "FR_.*_joint",
                "RL_.*_joint",
                "RR_.*_joint",
            ],
            stiffness=20.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    morph_options={},
)
