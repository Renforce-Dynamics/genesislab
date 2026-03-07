"""Joint-space action term implementations for GenesisLab.

These classes follow the same high-level structure as IsaacLab's joint
actions, but drive Genesis through the existing actuator / binding layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesislab.components.actuators import ActuatorBase, ImplicitActuator
from genesislab.managers.action_manager import ActionTerm
from genesislab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from .actions_cfg import JointPositionActionCfg


class JointPositionAction(ActionTerm):
    """Action term that maps normalized actions to joint position targets.

    Supports both implicit and explicit actuators:

    - Implicit actuators: sets position targets directly
    - Explicit actuators: computes torques via actuator.compute() and applies them
      via force control.
    """

    cfg: "JointPositionActionCfg"

    def __init__(self, cfg: "JointPositionActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)

        # Resolve entity.
        self._entity_name = cfg.asset_name
        entity = env.entities[self._entity_name]

        # Check if actuators are configured for this entity.
        self._actuators: dict[str, ActuatorBase] = {}
        self._has_explicit_actuators = False
        if hasattr(env, "_binding") and hasattr(env._binding, "_actuators"):
            entity_actuators = env._binding._actuators.get(self._entity_name, {})
            if entity_actuators:
                self._actuators = entity_actuators
                self._has_explicit_actuators = any(
                    not isinstance(act, ImplicitActuator) for act in self._actuators.values()
                )

        # Infer action dimension:
        # 1) sum of actuator joint counts if actuators exist
        # 2) otherwise, all DOFs on the entity.
        if self._actuators:
            self._action_dim = sum(actuator.num_joints for actuator in self._actuators.values())
        else:
            dof_pos = entity.data.joint_pos
            self._action_dim = dof_pos.shape[-1]

        # Buffers.
        self._raw_action = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._targets = torch.zeros_like(self._raw_action)

        # Offset.
        self._offset = torch.zeros_like(self._targets)
        if cfg.use_default_offset:
            default_joint_pos = entity.data.default_joint_pos
            if default_joint_pos.shape[-1] >= self._action_dim:
                self._offset[:] = default_joint_pos[:, :self._action_dim].clone()
            else:
                self._offset[:, :default_joint_pos.shape[-1]] = default_joint_pos.clone()

        # Scale.
        self._scale = float(cfg.scale)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        """Store and scale raw actions into joint position targets."""
        if actions.shape != self._raw_action.shape:
            if actions.shape[-1] == self._action_dim and actions.shape[0] == 1:
                actions = actions.expand_as(self._raw_action)
            else:
                raise ValueError(
                    f"Invalid action shape for JointPositionAction: expected "
                    f"{self._raw_action.shape}, got {actions.shape}."
                )

        self._raw_action[:] = actions
        self._targets[:] = self._offset + self._scale * actions

    def apply_actions(self) -> None:
        """Apply position targets or torques to Genesis."""
        if self._has_explicit_actuators:
            # Explicit actuators: compute torques and apply.
            entity = self._env.entities[self._entity_name]
            joint_pos = entity.data.joint_pos
            joint_vel = entity.data.joint_vel

            total_torques = torch.zeros_like(joint_pos)

            for actuator in self._actuators.values():
                if isinstance(actuator, ImplicitActuator):
                    continue

                if actuator.joint_indices == slice(None):
                    num_act_joints = actuator.num_joints
                    if joint_pos.shape[-1] >= num_act_joints:
                        act_joint_pos = joint_pos[:, :num_act_joints]
                        act_joint_vel = joint_vel[:, :num_act_joints]
                        act_targets = self._targets[:, :num_act_joints]
                    else:
                        act_joint_pos = torch.zeros(
                            joint_pos.shape[0], num_act_joints, dtype=joint_pos.dtype, device=joint_pos.device
                        )
                        act_joint_pos[:, :joint_pos.shape[-1]] = joint_pos
                        act_joint_vel = torch.zeros(
                            joint_vel.shape[0], num_act_joints, dtype=joint_vel.dtype, device=joint_vel.device
                        )
                        act_joint_vel[:, :joint_vel.shape[-1]] = joint_vel
                        act_targets = torch.zeros(
                            self._targets.shape[0], num_act_joints, dtype=self._targets.dtype, device=self._targets.device
                        )
                        act_targets[:, :self._targets.shape[-1]] = self._targets
                    joint_indices = slice(None)
                else:
                    if isinstance(actuator.joint_indices, torch.Tensor):
                        joint_indices = actuator.joint_indices.cpu().tolist()
                    else:
                        joint_indices = list(range(len(actuator.joint_names)))
                    act_joint_pos = joint_pos[:, joint_indices]
                    act_joint_vel = joint_vel[:, joint_indices]
                    act_targets = self._targets[:, joint_indices]

                control_action = ArticulationActions(
                    joint_positions=act_targets,
                    joint_velocities=None,
                    joint_efforts=None,
                    joint_indices=actuator.joint_indices,
                )

                control_action = actuator.compute(
                    control_action,
                    joint_pos=act_joint_pos,
                    joint_vel=act_joint_vel,
                )

                if control_action.joint_efforts is not None:
                    if actuator.joint_indices == slice(None):
                        num_act_joints = actuator.num_joints
                        if total_torques.shape[-1] >= num_act_joints:
                            total_torques[:, :num_act_joints] = control_action.joint_efforts
                        else:
                            total_torques[:] = control_action.joint_efforts[:, : total_torques.shape[-1]]
                    else:
                        if isinstance(actuator.joint_indices, torch.Tensor):
                            joint_indices_tensor = actuator.joint_indices
                        else:
                            joint_indices_tensor = torch.tensor(joint_indices, dtype=torch.long, device=self.device)
                        total_torques[:, joint_indices_tensor] = control_action.joint_efforts

            self._env.set_joint_targets(self._entity_name, total_torques, control_type="torque")
        else:
            # Implicit actuators / PD: set desired positions directly.
            self._env.set_joint_targets(self._entity_name, self._targets, control_type="position")

