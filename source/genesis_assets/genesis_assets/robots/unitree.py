"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Try to import robotlib asset path, but allow it to be None if not available
try:
    from robotlib import ROBOTLIB_ASSETLIB_DIR as ASSET_DIR
    from robotlib.unitree_rl_lab import UNITREE_MODEL_DIR
except ImportError:
    ASSET_DIR = None
    UNITREE_MODEL_DIR = None

##
# Configuration
##

# Go2 quadruped robot
UNITREE_GO2_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd" if UNITREE_MODEL_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.4],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={},
)

# Go2W quadruped robot (with wheels)
UNITREE_GO2W_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{UNITREE_MODEL_DIR}/Go2W/usd/go2w.usd" if UNITREE_MODEL_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.45],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={},
)

# B2 quadruped robot
UNITREE_B2_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{UNITREE_MODEL_DIR}/B2/usd/b2.usd" if UNITREE_MODEL_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.58],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={},
)

# H1 humanoid robot
UNITREE_H1_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{UNITREE_MODEL_DIR}/H1/h1/usd/h1.usd" if UNITREE_MODEL_DIR else "",
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

# G1 23DOF humanoid robot
UNITREE_G1_23DOF_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{UNITREE_MODEL_DIR}/G1/23dof/usd/g1_23dof_rev_1_0/g1_23dof_rev_1_0.usd" if UNITREE_MODEL_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.8],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={},
)
