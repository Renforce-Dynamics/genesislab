"""Booster humanoid robot configurations.

This package provides configurations for Booster humanoid robots:
- K1: 22DOF humanoid robot (URDF)
- T1: 23DOF humanoid robot (URDF)
- K1 Serial: 22DOF humanoid robot (USD)
"""

from .k1 import BOOSTER_K1_CFG
from .k1serial import BOOSTER_K1SERIAL_22DOF_CFG
from .t1 import BOOSTER_T1_CFG

__all__ = [
    "BOOSTER_K1_CFG",
    "BOOSTER_T1_CFG",
    "BOOSTER_K1SERIAL_22DOF_CFG",
]
