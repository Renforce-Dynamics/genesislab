"""Unitree G1 humanoid robot configurations.

This package provides configurations for the Unitree G1 humanoid robot:
- BeyondMimic: URDF variant with PD parameters
"""

from .beyondmimic import G1_BEYONDMIMIC_CFG

__all__ = [
    "G1_BEYONDMIMIC_CFG",
]
