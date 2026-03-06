"""Configuration for PI humanoid robots.

Reference: PI humanoid robot configurations.
"""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Try to import robotlib asset path, but allow it to be None if not available
try:
    from robotlib import ROBOTLIB_USD_DIR
except ImportError:
    ROBOTLIB_USD_DIR = None

##
# Configuration
##

PI_PLUS_25DOF_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ROBOTLIB_USD_DIR}/pi_plus_25dof/pi_plus_25dof.usd" if ROBOTLIB_USD_DIR else "",
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

PI_PLUS_27DOF_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ROBOTLIB_USD_DIR}/pi_plus_27dof/pi_plus_27dof.usd" if ROBOTLIB_USD_DIR else "",
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
