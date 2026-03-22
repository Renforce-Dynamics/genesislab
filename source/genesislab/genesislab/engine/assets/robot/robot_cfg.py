"""Robot configuration for GenesisLab."""

from __future__ import annotations

from dataclasses import MISSING

from genesislab.utils.configclass import configclass
from genesislab.components.actuators import ActuatorBaseCfg

from genesislab.engine.assets.articulation import ArticulationCfg, InitialPoseCfg

@configclass
class RobotCfg(ArticulationCfg):
    """Configuration for a robot entity to be added to a Genesis scene.

    This config is focused on rigid-body robots loaded from URDF/MJCF/USD.
    """
    name: str = "robot"

    fixed_base: bool = False
    """Whether the robot base is fixed (non-floating)."""

    # Actuator configuration (IsaacLab-style)
    actuators: dict[str, ActuatorBaseCfg] = None
    """Actuator configurations for the robot, similar to IsaacLab's ArticulationCfg.

    This is a dictionary mapping actuator group names to actuator configuration objects
    (e.g., `IdealPDActuatorCfg`, `ImplicitActuatorCfg`). Each actuator configuration
    specifies which joints it controls (via `joint_names_expr`) and the actuator parameters.

    All actuators compute torques explicitly and apply them via `control_dofs_force()`.
    The Genesis engine's internal PD gains (kp/kv) are set to 0.

    Example:
        ```python
        from genesislab.components.actuators import IdealPDActuatorCfg

        actuators={
            "default": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=100.0,
                damping=10.0,
            )
        }
        ```
    """

    # Default joint positions (for reset and action offset)
    default_joint_pos: dict[str, float] = None
    """Default joint positions for reset and action offset.
    
    This is a dictionary mapping joint name patterns (regex) to default position values.
    Used during reset to set initial joint positions, and by action managers when
    `use_default_offset=True` to compute action offsets.
    
    Example:
        ```python
        default_joint_pos={
            ".*_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        }
        ```
    """

