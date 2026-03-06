"""Unitree robot configurations.

This package provides configurations for various Unitree robots including:
- Go2: Quadruped robot
- Go2W: Quadruped robot with wheels
- B2: Quadruped robot
- H1: Humanoid robot
- G1 23DOF: Humanoid robot (23 degrees of freedom variant)
"""

from .b2 import UNITREE_B2_CFG
from .g1_23dof import UNITREE_G1_23DOF_CFG
from .go2 import UNITREE_GO2_CFG
from .go2w import UNITREE_GO2W_CFG
from .h1 import UNITREE_H1_CFG

__all__ = [
    "UNITREE_GO2_CFG",
    "UNITREE_GO2W_CFG",
    "UNITREE_B2_CFG",
    "UNITREE_H1_CFG",
    "UNITREE_G1_23DOF_CFG",
]
