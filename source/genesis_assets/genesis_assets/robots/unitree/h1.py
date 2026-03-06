"""Configuration for Unitree H1 humanoid robot.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_UNITREE_MODEL_DIR as UNITREE_MODEL_DIR

##
# Configuration
##

UNITREE_H1_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{UNITREE_MODEL_DIR}/H1/h1/usd/h1.usd",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 1.1],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={},
)
