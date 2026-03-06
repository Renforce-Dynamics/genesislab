"""Robot configurations for GenesisLab.

This package provides robot configurations aligned with robotlib/robotlib implementations.
All configurations use the GenesisLab RobotCfg format and can be used directly in scene configurations.

Available robots:
    - G1: Unitree G1 humanoid robot (opensource URDF and USD variants)
    - SMPL: SMPL humanoid robot
    - SMPLX: SMPLX humanoid robot
    - Booster: K1, T1, and K1 Serial humanoid robots
    - Unitree: Go2, Go2W, B2, H1, and G1 23DOF robots
    - PI: PI Plus 25DOF and 27DOF humanoid robots
    - R2: R2 wholebody humanoid robot
"""

from .booster import BOOSTER_K1_CFG, BOOSTER_K1SERIAL_22DOF_CFG, BOOSTER_T1_CFG
from .g1 import G1_CYLINDER_CFG, G1_OPENSOURCE_CFG
from .pi import PI_PLUS_25DOF_CFG, PI_PLUS_27DOF_CFG
from .r2 import R2_WHOLEBODY_CFG
from .smpl import SMPL_HUMANOID_CFG
from .smplx import SMPLX_HUMANOID_CFG
from .unitree import (
    UNITREE_B2_CFG,
    UNITREE_G1_23DOF_CFG,
    UNITREE_GO2_CFG,
    UNITREE_GO2W_CFG,
    UNITREE_H1_CFG,
)

__all__ = [
    # G1 robots
    "G1_OPENSOURCE_CFG",
    "G1_CYLINDER_CFG",
    # SMPL
    "SMPL_HUMANOID_CFG",
    # SMPLX
    "SMPLX_HUMANOID_CFG",
    # Booster robots
    "BOOSTER_K1_CFG",
    "BOOSTER_K1SERIAL_22DOF_CFG",
    "BOOSTER_T1_CFG",
    # Unitree robots
    "UNITREE_GO2_CFG",
    "UNITREE_GO2W_CFG",
    "UNITREE_B2_CFG",
    "UNITREE_H1_CFG",
    "UNITREE_G1_23DOF_CFG",
    # PI robots
    "PI_PLUS_25DOF_CFG",
    "PI_PLUS_27DOF_CFG",
    # R2 robot
    "R2_WHOLEBODY_CFG",
]
