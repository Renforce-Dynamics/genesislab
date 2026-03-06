"""Robot configurations for GenesisLab.

This package provides robot configurations for GenesisLab.
All configurations use the GenesisLab RobotCfg format and can be used directly in scene configurations.

The package is organized by robot manufacturer/brand:
    - unitree: Unitree robots (Go2, Go2W, B2, H1, G1 23DOF)
    - g1: Unitree G1 humanoid robot variants (beyondmimic URDF)
    - booster: Booster humanoid robots (K1, T1, K1 Serial)
    - smpl: SMPL and SMPLX humanoid robots
"""

# Import from subpackages
from .booster import BOOSTER_K1_CFG, BOOSTER_K1SERIAL_22DOF_CFG, BOOSTER_T1_CFG
from .g1 import G1_BEYONDMIMIC_CFG
from .smpl import SMPL_HUMANOID_CFG, SMPLX_HUMANOID_CFG
from .unitree import (
    UNITREE_B2_CFG,
    UNITREE_G1_23DOF_CFG,
    UNITREE_GO2_CFG,
    UNITREE_GO2W_CFG,
    UNITREE_H1_CFG,
)

__all__ = [
    # G1 robots
    "G1_BEYONDMIMIC_CFG",
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
]
