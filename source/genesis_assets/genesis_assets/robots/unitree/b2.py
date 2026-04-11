"""Configuration for Unitree B2 quadruped robot.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

from __future__ import annotations

from genesislab.components.actuators import IdealPDActuatorCfg
from genesislab.engine.assets.robot import InitialPoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_UNITREE_MODEL_DIR as UNITREE_MODEL_DIR

##
# Configuration
##

UNITREE_B2_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{UNITREE_MODEL_DIR}/B2/usd/b2.usd",
    initial_pose=InitialPoseCfg(
        pos=[0.0, 0.0, 0.58],
        quat=[1.0, 0.0, 0.0, 0.0]  # wxyz: identity,
    ),
    fixed_base=False,
    control_dofs=None,
    actuators={
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit=200,
            velocity_limit=23,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
        "2": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit=320,
            velocity_limit=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
    },
    morph_options={},
)
