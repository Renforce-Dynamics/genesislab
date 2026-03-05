"""Action terms for Go2 velocity tracking task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genesislab.managers.action_manager import ActionTerm, ActionTermCfg
from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedGenesisEnv


@configclass
class Go2ActionTermCfg(ActionTermCfg):
    """Configuration for Go2 joint position action term."""

    def build(self, env: "ManagerBasedGenesisEnv") -> "Go2ActionTerm":
        """Build the Go2 action term from this config."""
        return Go2ActionTerm(cfg=self, env=env)


class Go2ActionTerm(ActionTerm):
    """Action term that maps normalized actions to Go2 joint position targets."""

    def __init__(self, cfg: Go2ActionTermCfg, env: "ManagerBasedGenesisEnv"):
        super().__init__(cfg, env)
        # Infer action dimension from the controlled entity's DOFs.
        dof_pos, _ = env._binding.get_joint_state(cfg.entity_name)
        self._action_dim = dof_pos.shape[-1]

        self._raw_action = torch.zeros(
            (self.num_envs, self._action_dim), device=self.device
        )
        self._targets = torch.zeros_like(self._raw_action)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:
        """Store and scale raw actions into joint position targets.

        Actions are expected in [-1, 1] and are scaled to a small joint range.
        """
        if actions.shape != self._raw_action.shape:
            if actions.shape[-1] == self._action_dim and actions.shape[0] == 1:
                actions = actions.expand_as(self._raw_action)
            else:
                raise ValueError(
                    f"Invalid action shape for Go2ActionTerm: expected "
                    f"{self._raw_action.shape}, got {actions.shape}."
                )

        self._raw_action[:] = actions
        # Simple scaling – in practice this should respect joint limits.
        self._targets[:] = actions * 0.5

    def apply_actions(self) -> None:
        """Write joint position targets to the Genesis binding."""
        self._env._binding.set_joint_targets(
            self.cfg.entity_name,
            self._targets,
            control_type="position",
        )

