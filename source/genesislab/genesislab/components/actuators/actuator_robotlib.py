"""Robotlib-specific actuator implementations.

This module provides actuator models used in robotlib configurations, including:
- DelayedImplicitActuator: Implicit actuator with delayed command application
- UnitreeActuator: Unitree-specific actuator with torque-speed curve
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from genesislab.components.additional.buffers import DelayBuffer
from genesislab.utils.configclass import configclass
from genesislab.utils.types import ArticulationActions

from .actuator_base import ActuatorBase
from .actuator_pd import IdealPDActuator, ImplicitActuator
from .actuator_pd_cfg import IdealPDActuatorCfg, ImplicitActuatorCfg

if TYPE_CHECKING:
    from .actuator_base_cfg import ActuatorBaseCfg


class DelayedImplicitActuator(ImplicitActuator):
    """Implicit PD actuator with delayed command application.

    This class extends the :class:`ImplicitActuator` class by adding a delay to the actuator commands.
    The delay is implemented using a delay buffer that stores the actuator commands for a certain
    number of physics steps. The most recent actuation value is pushed to the buffer at every physics
    step, but the final actuation value applied to the simulation is lagged by a certain number of
    physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and
    maximum time lag bounds at every reset. The minimum and maximum time lag values are set in the
    configuration instance passed to the class.
    """

    cfg: "DelayedImplicitActuatorCfg"
    """The configuration for the actuator model."""

    def __init__(self, cfg: "DelayedImplicitActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # Instantiate delay buffers using IsaacLab-style interface
        # Note: We use a wrapper to adapt genesislab's DelayBuffer to IsaacLab's interface
        self.positions_delay_buffer = _IsaacLabStyleDelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = _IsaacLabStyleDelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = _IsaacLabStyleDelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # All of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # Number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # Set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # Set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # Reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Apply delay based on the delay model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # Compute actuator model
        return super().compute(control_action, joint_pos, joint_vel)


class UnitreeActuator(IdealPDActuator):
    """Unitree actuator class that implements a torque-speed curve for the actuators.

    The torque-speed curve is defined as follows:

            Torque, N·m
                ^
    Y2──────────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
    ------------+--------------|------> velocity: rad/s
                              X1   X2

    Y1: Peak Torque Test (Torque and Speed in the Same Direction)
    Y2: Peak Torque Test (Torque and Speed in the Opposite Direction)
    X1: Maximum Speed at Full Torque (T-N Curve Knee Point)
    X2: No-Load Speed Test
    """

    cfg: "UnitreeActuatorCfg"

    def __init__(self, cfg: "UnitreeActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        assert cfg.X1 < cfg.X2, "X1 must be less than X2"
        assert cfg.Y1 <= cfg.Y2, "Y1 must be less than or equal to Y2"

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = torch.tensor(cfg.Y1, dtype=self.computed_effort.dtype, device=self.computed_effort.device)
        self._effort_y2 = torch.tensor(cfg.Y2, dtype=self.computed_effort.dtype, device=self.computed_effort.device)
        self._velocity_x1 = torch.tensor(cfg.X1, dtype=self.computed_effort.dtype, device=self.computed_effort.device)
        self._velocity_x2 = torch.tensor(cfg.X2, dtype=self.computed_effort.dtype, device=self.computed_effort.device)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Save current joint vel (ensure shape matches)
        if joint_vel.shape == self._joint_vel.shape:
            self._joint_vel[:] = joint_vel
        else:
            # Reshape or slice to match actuator's joint count
            if joint_vel.shape[-1] > self._joint_vel.shape[-1]:
                # If joint_vel has more DOFs, take only the first num_joints
                self._joint_vel[:] = joint_vel[:, :self._joint_vel.shape[-1]]
            else:
                # If joint_vel has fewer DOFs, pad with zeros or expand
                self._joint_vel[:, :joint_vel.shape[-1]] = joint_vel
                self._joint_vel[:, joint_vel.shape[-1]:] = 0.0
        # Calculate the desired joint torques
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # Check if the effort is the same direction as the joint velocity
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        # Check if the joint velocity is less than the max speed at full torque
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, max_effort, self._compute_effort_limit(max_effort)
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort):
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


class _IsaacLabStyleDelayBuffer:
    """Wrapper to adapt genesislab's DelayBuffer to IsaacLab's DelayBuffer interface.

    This wrapper provides the IsaacLab-style interface (history_length, batch_size, device)
    and methods (set_time_lag, compute(data)) while using genesislab's DelayBuffer internally.
    """

    def __init__(self, history_length: int, batch_size: int, device: str):
        """Initialize the delay buffer wrapper.

        Args:
            history_length: Maximum delay in time steps.
            batch_size: Number of parallel environments.
            device: Device for tensor operations.
        """
        self._history_length = history_length
        self._batch_size = batch_size
        self._device = device

        # Use genesislab's DelayBuffer with min_lag=0, max_lag=history_length
        self._buffer = DelayBuffer(
            min_lag=0,
            max_lag=history_length,
            batch_size=batch_size,
            device=device,
            per_env=True,  # Each environment has its own lag
            hold_prob=0.0,  # No hold probability
            update_period=0,  # Update every step
            per_env_phase=False,  # No phase offset needed
        )

    def set_time_lag(self, time_lag: int | torch.Tensor, batch_ids: Sequence[int] | None = None):
        """Set the time lag for the delay buffer.

        Args:
            time_lag: The desired delay (int or tensor).
            batch_ids: Batch indices to set, or None for all.
        """
        if isinstance(time_lag, int):
            # If int, create a tensor with the same value for all batch indices
            if batch_ids is None:
                lags = torch.full((self._batch_size,), time_lag, dtype=torch.long, device=self._device)
            else:
                if isinstance(batch_ids, slice):
                    batch_ids = list(range(*batch_ids.indices(self._batch_size)))
                lags = torch.full((len(batch_ids),), time_lag, dtype=torch.long, device=self._device)
        else:
            # If tensor, use it directly
            lags = time_lag.to(device=self._device, dtype=torch.long)

        # Clamp to valid range
        lags = lags.clamp(0, self._history_length)
        self._buffer.set_lags(lags, batch_ids)

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the delay buffer.

        Args:
            batch_ids: Batch indices to reset, or None for all.
        """
        self._buffer.reset(batch_ids)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Append data and return delayed version.

        Args:
            data: Input data tensor of shape (batch_size, ...).

        Returns:
            Delayed data tensor of the same shape.
        """
        self._buffer.append(data)
        return self._buffer.compute()


"""
Configuration classes.
"""


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed implicit actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


@configclass
class UnitreeActuatorCfg(IdealPDActuatorCfg):
    """Configuration for Unitree actuators."""

    class_type: type = UnitreeActuator

    X1: float = MISSING
    """Maximum Speed at Full Torque (T-N Curve Knee Point) Unit: rad/s"""

    X2: float = MISSING
    """No-Load Speed Test Unit: rad/s"""

    Y1: float = MISSING
    """Peak Torque Test (Torque and Speed in the Same Direction) Unit: N*m"""

    Y2: float = MISSING
    """Peak Torque Test (Torque and Speed in the Opposite Direction) Unit: N*m"""


# Predefined Unitree actuator configurations
@configclass
class UnitreeActuatorCfg_M107_15(UnitreeActuatorCfg):
    """Unitree M107-15 actuator configuration."""

    X1 = 14.0
    X2 = 25.6
    Y1 = 150.0
    Y2 = 102.8


@configclass
class UnitreeActuatorCfg_M107_24(UnitreeActuatorCfg):
    """Unitree M107-24 actuator configuration."""

    X1 = 8.8
    X2 = 16
    Y1 = 240
    Y2 = 292.5


@configclass
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    """Unitree Go2HV actuator configuration."""

    X1 = 13.5
    X2 = 30
    Y1 = 20.2
    Y2 = 23.4


@configclass
class UnitreeActuatorCfg_N7520_14p3(UnitreeActuatorCfg):
    """Unitree N7520-14.3 actuator configuration."""

    X1 = 22.63
    X2 = 35.52
    Y1 = 71
    Y2 = 83.3


@configclass
class UnitreeActuatorCfg_N7520_22p5(UnitreeActuatorCfg):
    """Unitree N7520-22.5 actuator configuration."""

    X1 = 14.5
    X2 = 22.7
    Y1 = 111.0
    Y2 = 131.0


@configclass
class UnitreeActuatorCfg_N5010_16(UnitreeActuatorCfg):
    """Unitree N5010-16 actuator configuration."""

    X1 = 27.0
    X2 = 41.5
    Y1 = 9.5
    Y2 = 17.0


@configclass
class UnitreeActuatorCfg_N5020_16(UnitreeActuatorCfg):
    """Unitree N5020-16 actuator configuration."""

    X1 = 30.86
    X2 = 40.13
    Y1 = 24.8
    Y2 = 31.9


@configclass
class UnitreeActuatorCfg_W4010_25(UnitreeActuatorCfg):
    """Unitree W4010-25 actuator configuration."""

    X1 = 15.3
    X2 = 24.76
    Y1 = 4.8
    Y2 = 8.6
