"""Control methods for GenesisBinding."""

from __future__ import annotations

from typing import Any

import torch


class Controller:
    """Helper class for controlling entities."""

    def __init__(self, binding: Any):
        """Initialize the controller.

        Args:
            binding: Reference to the GenesisBinding instance.
        """
        self._binding = binding

    def set_joint_targets(self, entity_name: str, targets: torch.Tensor, control_type: str = "position") -> None:
        """Set joint control targets for an entity.

        Args:
            entity_name: Name of the entity.
            targets: Target values of shape (num_envs, num_dofs).
            control_type: Type of control ('position', 'velocity', or 'torque').
        """
        entity = self._binding._entities[entity_name]
        dof_indices = self._binding._dof_indices.get(entity_name)

        if control_type == "position":
            if dof_indices is not None:
                entity.control_dofs_position(targets, dof_indices)
            else:
                entity.control_dofs_position(targets)
        elif control_type == "velocity":
            if dof_indices is not None:
                entity.control_dofs_velocity(targets, dof_indices)
            else:
                entity.control_dofs_velocity(targets)
        elif control_type == "torque":
            # Torque control: apply forces directly
            # Genesis uses control_dofs_force() for torque/force control
            if dof_indices is not None:
                entity.control_dofs_force(targets, dof_indices)
            else:
                entity.control_dofs_force(targets)
        else:
            raise ValueError(f"Unknown control type: {control_type}")

    def set_pd_gains(self, entity_name: str, kp: torch.Tensor, kd: torch.Tensor) -> None:
        """Set PD gains for an entity's joints.

        Args:
            entity_name: Name of the entity.
            kp: Position gains of shape (num_dofs,) or (num_envs, num_dofs).
            kd: Velocity gains of shape (num_dofs,) or (num_envs, num_dofs).
        """
        entity = self._binding._entities[entity_name]
        dof_indices = self._binding._dof_indices.get(entity_name)

        if dof_indices is not None:
            entity.set_dofs_kp(kp, dof_indices)
            entity.set_dofs_kv(kd, dof_indices)
        else:
            entity.set_dofs_kp(kp)
            entity.set_dofs_kv(kd)
