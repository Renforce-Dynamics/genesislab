"""Common command terms for velocity tracking locomotion tasks.

These command terms can be used to define commands in the MDP configuration.
They follow the same interface as IsaacLab's command terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from genesislab.managers.command_manager import CommandTerm, CommandTermCfg
from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedGenesisEnv


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator.

    This command generator samples velocity commands uniformly from specified ranges.
    """

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False."""

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command.
    Defaults to 1.0. Only used if heading_command is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.
        Only used if heading_command is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    def build(self, env: "ManagerBasedGenesisEnv") -> "UniformVelocityCommand":
        """Build the uniform velocity command term from this config."""
        return UniformVelocityCommand(cfg=self, env=env)


class UniformVelocityCommand(CommandTerm):
    """Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.
    """

    def __init__(self, cfg: UniformVelocityCommandCfg, env: "ManagerBasedGenesisEnv"):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)

        # Check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the "
                "`ranges.heading` parameter is set to None."
            )

        # Create buffers to store the command
        # Command: [x vel, y vel, yaw vel]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        # Metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self) -> None:
        """Update metrics based on current state."""
        # Get robot entity
        entity = self._env.entities[self.cfg.asset_name]
        
        # Time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        
        # Get body frame velocities
        lin_vel_b = entity.data.root_lin_vel_b if hasattr(entity.data, "root_lin_vel_b") else entity.data.root_lin_vel_w
        ang_vel_b = entity.data.root_ang_vel_b if hasattr(entity.data, "root_ang_vel_b") else entity.data.root_ang_vel_w
        
        # Update metrics
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample velocity command for specified environments.

        Args:
            env_ids: Environment indices to resample.
        """
        if len(env_ids) == 0:
            return

        # Sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        
        # Linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # Linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # Angular velocity - yaw (rotation around z)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        
        # Heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # Update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        # Update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self) -> None:
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # Get robot entity
            entity = self._env.entities[self.cfg.asset_name]
            quat = entity.data.root_quat_w
            
            # Extract yaw from quaternion (simplified - assumes [x, y, z, w] format)
            # For proper yaw extraction, we'd need quaternion to euler conversion
            # For now, use a simple approximation
            if quat.shape[-1] == 4:
                # Approximate yaw from quaternion (this is simplified)
                # In practice, use proper quaternion to euler conversion
                yaw_current = torch.atan2(2 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]),
                                         1 - 2 * (quat[:, 1]**2 + quat[:, 2]**2))
            else:
                yaw_current = torch.zeros(self.num_envs, device=self.device)
            
            # Compute heading error and convert to angular velocity
            heading_error = self.heading_target - yaw_current
            # Wrap to [-pi, pi]
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
            
            # Update angular velocity for heading environments
            self.vel_command_b[self.is_heading_env, 2] = (
                self.cfg.heading_control_stiffness * heading_error[self.is_heading_env]
            )
        
        # Set velocity to zero for standing environments
        self.vel_command_b[self.is_standing_env, :] = 0.0

