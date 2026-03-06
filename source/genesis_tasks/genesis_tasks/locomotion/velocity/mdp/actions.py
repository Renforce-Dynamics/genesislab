"""Common action terms for velocity tracking locomotion tasks.

These action terms can be used to define actions in the MDP configuration.
They follow the same interface as IsaacLab's action terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesislab.components.actuators import ActuatorBase, ImplicitActuator
from genesislab.managers.action_manager import ActionTerm, ActionTermCfg
from genesislab.utils.configclass import configclass
from genesislab.utils.types import ArticulationActions

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
        """Build the joint position action term from this config.

        Note:
            The base :class:`ActionTerm` expects the ``entity_name`` field to be
            populated in :class:`ActionTermCfg`. For locomotion tasks we use the
            more semantically clear ``asset_name`` field instead. To keep
            compatibility with the base manager code, we mirror ``asset_name``
            into ``entity_name`` here before constructing the term.
        """
        # Ensure the base config field used by :class:`ActionTerm` is set.
        self.entity_name = self.asset_name
        return JointPositionAction(cfg=self, env=env)


class JointPositionAction(ActionTerm):
    """Action term that maps normalized actions to joint position targets.

    Supports both implicit and explicit actuators:
    - Implicit actuators: Sets position targets directly (engine handles PD control)
    - Explicit actuators: Computes torques via actuator.compute() and applies via control_dofs_force()
    """

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

        # Check if actuators are configured for this entity
        self._actuators: dict[str, ActuatorBase] = {}
        if hasattr(env, "_binding") and hasattr(env._binding, "_actuators"):
            entity_actuators = env._binding._actuators.get(self._entity_name, {})
            if entity_actuators:
                self._actuators = entity_actuators
                # Check if we have explicit actuators
                self._has_explicit_actuators = any(
                    not isinstance(act, ImplicitActuator) for act in self._actuators.values()
                )
            else:
                self._has_explicit_actuators = False
        else:
            self._has_explicit_actuators = False

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
        """Write joint position targets or computed torques to the simulation.

        If explicit actuators are configured, computes torques via actuator.compute()
        and applies them via control_dofs_force(). Otherwise, sets position targets
        directly (for implicit actuators or legacy PD control).
        """
        if self._has_explicit_actuators:
            # For explicit actuators: compute torques and apply via force control
            # Get current joint state
            entity = self._env.entities[self._entity_name]
            joint_pos = entity.data.joint_pos
            joint_vel = entity.data.joint_vel

            # Aggregate torques from all actuators
            # Initialize with zeros - each actuator will fill in its portion
            total_torques = torch.zeros_like(joint_pos)

            for actuator_name, actuator in self._actuators.items():
                # Skip implicit actuators (they use position control)
                if isinstance(actuator, ImplicitActuator):
                    continue

                # Get joint indices for this actuator
                if actuator.joint_indices == slice(None):
                    # All joints
                    actuator_joint_pos = joint_pos
                    actuator_joint_vel = joint_vel
                    actuator_targets = self._targets
                    joint_indices = slice(None)
                else:
                    # Subset of joints
                    if isinstance(actuator.joint_indices, torch.Tensor):
                        joint_indices = actuator.joint_indices.cpu().tolist()
                    else:
                        joint_indices = list(range(len(actuator.joint_names)))
                    actuator_joint_pos = joint_pos[:, joint_indices]
                    actuator_joint_vel = joint_vel[:, joint_indices]
                    actuator_targets = self._targets[:, joint_indices]

                # Create control action
                control_action = ArticulationActions(
                    joint_positions=actuator_targets,
                    joint_velocities=None,  # No velocity target for position control
                    joint_efforts=None,  # No feed-forward torque
                    joint_indices=actuator.joint_indices,
                )

                # Compute torques via actuator model
                control_action = actuator.compute(
                    control_action,
                    joint_pos=actuator_joint_pos,
                    joint_vel=actuator_joint_vel,
                )

                # Apply computed torques to the appropriate joint indices
                if control_action.joint_efforts is not None:
                    if actuator.joint_indices == slice(None):
                        # All joints: replace all torques
                        total_torques[:] = control_action.joint_efforts
                    else:
                        # Subset of joints: set only those joints
                        if isinstance(actuator.joint_indices, torch.Tensor):
                            joint_indices_tensor = actuator.joint_indices
                        else:
                            joint_indices_tensor = torch.tensor(
                                joint_indices, dtype=torch.long, device=self.device
                            )
                        total_torques[:, joint_indices_tensor] = control_action.joint_efforts

            # Apply torques via force control
            self._env.set_joint_targets(
                self._entity_name,
                total_torques,
                control_type="torque",
            )
        else:
            # For implicit actuators or legacy PD control: set position targets directly
            self._env.set_joint_targets(
                self._entity_name,
                self._targets,
                control_type="position",
            )

