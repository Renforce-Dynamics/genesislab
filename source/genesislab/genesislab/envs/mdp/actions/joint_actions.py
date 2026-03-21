"""Joint-space action term implementations for GenesisLab.

These classes follow the same high-level structure as IsaacLab's joint
actions, but drive Genesis through the existing actuator / scene layer.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from genesislab.components.actuators import ActuatorBase, ArticulationActions
from genesislab.managers.action_manager import ActionTerm, ActionTermCfg
from genesislab.utils.configclass import configclass
from genesislab.utils.configclass.string import resolve_matching_names_values

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv

class JointPositionAction(ActionTerm):
    """Action term that maps normalized actions to joint position targets.

    Each entity's :class:`~genesislab.engine.scene.actuator_manager.ActuatorManager` sets
    engine joint ``kp``/``kv`` from actuator stiffness/damping. This term
    therefore always:

    1. Builds :class:`~genesislab.components.actuators.articulation_actions.ArticulationActions`
       with desired joint positions.
    2. Calls :meth:`~genesislab.components.actuators.actuator_base.ActuatorBase.compute`
       (PD or DC motor, etc.), which must populate ``control_action.joint_efforts``.
    3. Applies those torques via :meth:`~genesislab.components.actuators.actuator_base.ActuatorBase.apply_torques`.

    ``ImplicitActuator`` is aligned with this path (it fills ``joint_efforts`` like
    :class:`~genesislab.components.actuators.actuator_pd.IdealPDActuator`).
    """

    cfg: "JointPositionActionCfg"

    def __init__(self, cfg: "JointPositionActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._entity_name = cfg.entity_name
        entity = env.entities[self._entity_name]

        # Get the specific actuator for this action term
        if cfg.actuator_name not in entity.actuators:
            raise ValueError(
                f"Actuator '{cfg.actuator_name}' not found for entity '{self._entity_name}'. "
                f"Available actuators: {list(entity.actuators.keys())}"
            )
        self._actuator = entity.actuators[cfg.actuator_name]

        # Action dimension matches actuator's number of joints
        self._action_dim = self._actuator.num_joints

        # Buffers: action-space buffers (in actuator's joint order)
        self._raw_action = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._targets = torch.zeros((self.num_envs, self._action_dim), device=self.device)

        # Build offset: combine default joint pos and config offset
        self._offset = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        
        default_joint_pos_dof = entity.data.default_joint_pos
        actuator_joint_names = self._actuator.joint_names
        
        # Map default joint positions from DOF order to actuator's joint order
        if default_joint_pos_dof is not None:
            # Extract default positions for this actuator's DOFs
            default_joint_pos_actuator = default_joint_pos_dof[:, self._actuator.dof_indices]
            self._offset[:] = default_joint_pos_actuator
        
        # Apply config offset (if provided as dict, match to joint names)
        if cfg.offset != 0.0:
            if isinstance(cfg.offset, dict):
                # Match offset values to actuator's joint names
                _, _, offset_values = resolve_matching_names_values(
                    cfg.offset, actuator_joint_names, preserve_order=False
                )
                offset_tensor = torch.tensor(offset_values, dtype=self._offset.dtype, device=self.device)
                self._offset += offset_tensor.unsqueeze(0)  # Broadcast to (num_envs, num_joints)
            else:
                # Scalar offset: apply to all joints
                self._offset += float(cfg.offset)

        # Scale: handle dict or scalar
        if isinstance(cfg.scale, dict):
            # Match scale values to actuator's joint names
            _, _, scale_values = resolve_matching_names_values(
                cfg.scale, actuator_joint_names, preserve_order=False
            )
            self._scale = torch.tensor(scale_values, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            self._scale = float(cfg.scale)
        
        # Build clip bounds (cached in base class)
        self._build_clip_bounds(self._action_dim, cfg.clip, actuator_joint_names)

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
        
        # Apply scale (handle both scalar and tensor)
        if isinstance(self._scale, torch.Tensor):
            self._targets[:] = self._offset + self._scale * actions
        else:
            self._targets[:] = self._offset + self._scale * actions
        
        # Apply clipping using cached bounds from base class
        self._targets[:] = self._apply_clip(self._targets)

    def apply_actions(self) -> None:
        """PD torque from targets, then ``control_dofs_force`` on joint DOFs."""
        entity = self._env.entities[self._entity_name]
        raw_entity = entity.raw_entity
        joint_pos = entity.data.joint_pos
        joint_vel = entity.data.joint_vel
        
        # Get joint state for this actuator's DOFs (in robot DOF order)
        act_joint_pos = joint_pos[:, self._actuator.dof_indices]
        act_joint_vel = joint_vel[:, self._actuator.dof_indices]

        # Create control action with position targets (in actuator's joint order)
        control_action = ArticulationActions(
            joint_positions=self._targets,
            joint_velocities=None,
            joint_efforts=None,
            joint_indices=None,
        )
        # Compute torques (in actuator's joint order)
        control_action = self._actuator.compute(
            control_action,
            joint_pos=act_joint_pos,
            joint_vel=act_joint_vel,
        )
        # Apply torques directly using actuator's apply_torques method
        # This allows multiple actuators to apply independently
        assert control_action.joint_efforts is not None, (
            "actuator.compute() must set control_action.joint_efforts "
            "(JointPositionAction uses control_dofs_force / CTRL_MODE.FORCE). "
            "Check ImplicitActuator / IdealPDActuator implementation."
        )
        self._actuator.apply_torques(raw_entity, control_action.joint_efforts)

@configclass
class JointActionCfg(ActionTermCfg):
    joint_names: list[str] = MISSING
    scale: float = 1.0
    offset: float = 0.0
    preserve_order: bool = False


@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for joint position action term."""
    class_type: type = JointPositionAction  # Will be set to JointPositionAction below
    """The action term class type. Set automatically."""
    actuator_name: str = MISSING
    """Name of the actuator to use for this action term."""
    use_default_offset: bool = True