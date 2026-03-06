# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different actuator models.

Actuator models are used to model the behavior of the actuators in an articulation. These
are usually meant to be used in simulation to model different actuator dynamics and delays.

There are two main categories of actuator models that are supported:

- **Implicit**: Motor model with ideal PD from the physics engine. This is similar to having a continuous time
  PD controller. The motor model is implicit in the sense that the motor model is not explicitly defined by the user.
- **Explicit**: Motor models based on physical drive models.

  - **Physics-based**: Derives the motor models based on first-principles.
  - **Neural Network-based**: Learned motor models from actuator data.

Every actuator model inherits from :class:`ActuatorBase`, which defines the common
interface for all actuator models.
"""

from .actuator_base import ActuatorBase
from .actuator_base_cfg import ActuatorBaseCfg
from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from .actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from .actuator_pd import DCMotor, DelayedPDActuator, IdealPDActuator, ImplicitActuator, RemotizedPDActuator
from .actuator_pd_cfg import (
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
    RemotizedPDActuatorCfg,
)
from .actuator_robotlib import (
    DelayedImplicitActuator,
    UnitreeActuator,
    DelayedImplicitActuatorCfg,
    UnitreeActuatorCfg,
    UnitreeActuatorCfg_Go2HV,
    UnitreeActuatorCfg_M107_15,
    UnitreeActuatorCfg_M107_24,
    UnitreeActuatorCfg_N5010_16,
    UnitreeActuatorCfg_N5020_16,
    UnitreeActuatorCfg_N7520_14p3,
    UnitreeActuatorCfg_N7520_22p5,
    UnitreeActuatorCfg_W4010_25,
)