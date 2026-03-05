"""Common action terms for velocity tracking locomotion tasks.

These action terms can be used to define actions in the MDP configuration.
They follow the same interface as IsaacLab's action terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesislab.managers.action_manager import ActionTerm, ActionTermCfg
from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedGenesisEnv


@configclass
class JointPositionActionCfg(ActionTermCfg):
    """Configuration for joint position action term.

    This action term applies normalized actions to joint position targets.
    """

    asset_name: str = "robot"
    """Name of the asset entity to control."""

    joint_names: list[str] = [".*"]
    """List of joint names or patterns to control. Defaults to all joints."""

    scale: float = 0.5
    """Scaling factor for actions. Defaults to 0.5."""

    use_default_offset: bool = True
    """Whether to use default joint positions as offset. Defaults to True."""

    def build(self, env: "ManagerBasedGenesisEnv") -> "JointPositionAction":
        """Build the joint position action term from this config."""
        return JointPositionAction(cfg=self, env=env)


class JointPositionAction(ActionTerm):
    """Action term that maps normalized actions to joint position targets."""

    def __init__(self, cfg: JointPositionActionCfg, env: "ManagerBasedGenesisEnv"):
        super().__init__(cfg, env)
        
        # Get entity
        self._entity_name = cfg.asset_name
        entity = env.entities[self._entity_name]
        
        # Infer action dimension from the controlled entity's DOFs
        dof_pos = entity.data.joint_pos
        self._action_dim = dof_pos.shape[-1]

        # Create buffers
        self._raw_action = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._targets = torch.zeros_like(self._raw_action)
        
        # Set default offset if requested
        self._offset = torch.zeros_like(self._targets)
        if cfg.use_default_offset:
            # Use current joint positions as default (can be improved with actual default positions)
            self._offset[:] = dof_pos.clone()
        
        # Set scale
        self._scale = cfg.scale

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        """Store and scale raw actions into joint position targets.

        Actions are expected in [-1, 1] and are scaled and offset.
        """
        if actions.shape != self._raw_action.shape:
            if actions.shape[-1] == self._action_dim and actions.shape[0] == 1:
                actions = actions.expand_as(self._raw_action)
            else:
                raise ValueError(
                    f"Invalid action shape for JointPositionAction: expected "
                    f"{self._raw_action.shape}, got {actions.shape}."
                )

        self._raw_action[:] = actions
        # Apply scale and offset: target = offset + scale * action
        self._targets[:] = self._offset + self._scale * actions

    def apply_actions(self) -> None:
        """Write joint position targets using the environment interface."""
        self._env.set_joint_targets(
            self._entity_name,
            self._targets,
            control_type="position",
        )

