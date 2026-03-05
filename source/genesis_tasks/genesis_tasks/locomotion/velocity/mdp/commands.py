"""Command terms for Go2 velocity tracking task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genesislab.managers.command_manager import CommandTerm, CommandTermCfg
from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedGenesisEnv


@configclass
class VelocityCommandCfg(CommandTermCfg):
    """Configuration for Go2 forward velocity command term."""

    velocity_range: tuple[float, float] = (0.0, 1.5)

    def build(self, env: "ManagerBasedGenesisEnv") -> "VelocityCommand":
        """Build the velocity command term from this config."""
        return VelocityCommand(cfg=self, env=env)


class VelocityCommand(CommandTerm):
    """Velocity command term for Go2.

    Generates forward velocity commands that are resampled at intervals.
    """

    def __init__(self, cfg: VelocityCommandCfg, env):
        """Initialize the velocity command term.

        Args:
            cfg: Command term configuration.
            env: Environment instance.
        """
        super().__init__(cfg, env)

        # Get velocity range from config
        self.velocity_range = cfg.velocity_range

        # Initialize command buffer
        self._command = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Current velocity command.

        Returns:
            Command tensor of shape (num_envs, 1).
        """
        return self._command

    def _update_metrics(self) -> None:
        """Update metrics based on current state."""
        # No metrics to update for simple velocity command
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Resample velocity command for specified environments.

        Args:
            env_ids: Environment indices to resample.
        """
        if len(env_ids) == 0:
            return

        # Sample random velocity in range
        v_min, v_max = self.velocity_range
        new_commands = torch.empty((len(env_ids), 1), device=self.device)
        new_commands.uniform_(v_min, v_max)

        self._command[env_ids] = new_commands

    def _update_command(self) -> None:
        """Update command based on current state."""
        # Commands are constant until resampled
        pass

