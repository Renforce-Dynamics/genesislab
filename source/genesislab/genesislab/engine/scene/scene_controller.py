"""Scene controller for control and state setting.

This module provides the SceneController class for controlling entities
and setting their states (reset, joint positions, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .lab_scene import LabScene


class SceneController:
    """Helper class for controlling entities and setting states."""
    
    def __init__(self, scene: "LabScene"):
        """Initialize the scene controller.
        
        Args:
            scene: Reference to the LabScene instance.
        """
        self._scene = scene
    
    def reset(self, env_ids: torch.Tensor = None) -> None:
        """Reset specified environments to initial state.
        
        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            self._scene.gs_scene.reset()
        else:
            # if isinstance(env_ids, torch.Tensor): env_ids = env_ids.cpu().tolist()
            self._scene.gs_scene.reset(envs_idx=env_ids)
    
    def step(self) -> None:
        """Step the physics simulation by one timestep."""
        self._scene.gs_scene.step()
    
    def set_joint_targets(
        self, entity_name: str, targets: torch.Tensor, control_type: str = "position"
    ) -> None:
        """Set joint control targets for an entity.
        
        Args:
            entity_name: Name of the entity.
            targets: Target values of shape (num_envs, num_dofs).
            control_type: Type of control ('position', 'velocity', or 'torque').
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity
        dof_indices = lab_entity.dof_indices
        
        if control_type == "position":
            entity.control_dofs_position(targets, dof_indices)
        elif control_type == "velocity":
            entity.control_dofs_velocity(targets, dof_indices)
        elif control_type == "torque":
            # Torque control: apply forces directly
            # Genesis uses control_dofs_force() for torque/force control
            entity.control_dofs_force(targets, dof_indices)

        else:
            raise ValueError(f"Unknown control type: {control_type}")
    
    def set_joint_positions(self, entity_name: str, positions: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Set joint positions for an entity (for reset/initialization).
        
        Args:
            entity_name: Name of the entity.
            positions: Joint positions of shape (num_envs, num_dofs) or (num_dofs,).
            env_ids: Environment indices to set. If None, sets all environments.
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity
        dof_indices = lab_entity.dof_indices

        if dof_indices is not None:
            entity.set_dofs_position(positions, dof_indices + 6, envs_idx=env_ids)
        else:
            num_joint_dofs = positions.shape[-1]
            entity.set_dofs_position(
                positions,
                dofs_idx_local=slice(6, 6 + num_joint_dofs),
                envs_idx=env_ids,
            )

    def set_joint_velocities(self, entity_name: str, velocities: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Set joint velocities for an entity (for reset/initialization).
        
        Args:
            entity_name: Name of the entity.
            velocities: Joint velocities of shape (num_envs, num_dofs) or (num_dofs,).
            env_ids: Environment indices to set. If None, sets all environments.
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity
        dof_indices = lab_entity.dof_indices

        if dof_indices is not None:
            entity.set_dofs_velocity(velocities, dof_indices, envs_idx=env_ids)
        else:
            num_joint_dofs = velocities.shape[-1]
            entity.set_dofs_velocity(
                velocities,
                dofs_idx_local=slice(6, 6 + num_joint_dofs),
                envs_idx=env_ids,
            )
    
    def set_root_state(
        self,
        entity_name: str,
        position: torch.Tensor = None,
        quaternion: torch.Tensor = None,
        linear_velocity: torch.Tensor = None,
        angular_velocity: torch.Tensor = None,
        env_ids: torch.Tensor = None,
    ) -> None:
        """Set root state for an entity (for reset/initialization).

        Args:
            entity_name: Name of the entity.
            position: Root position of shape (num_envs, 3) or (3,). If None, not set.
            quaternion: Root quaternion [w, x, y, z] (wxyz) of shape (num_envs, 4) or (4,).
                If None, not set. Must be in wxyz format matching Genesis API.
            linear_velocity: Linear velocity of shape (num_envs, 3) or (3,). If None, not set.
            angular_velocity: Angular velocity of shape (num_envs, 3) or (3,). If None, not set.
            env_ids: Environment indices to set. If None, sets all environments.
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity

        # Base pose: prefer setting through DOFs (first 7 generalized coordinates: pos_xyz + quat_wxyz)
        # when both position and quaternion are provided. This keeps reset logic consistent
        # with joint-level operations and avoids mixing root-state and DOF APIs.
        if position is not None:
            entity.set_pos(position, envs_idx=env_ids)
        if quaternion is not None:
            entity.set_quat(quaternion, envs_idx=env_ids)

        # Base velocities: use first 6 DOFs (lin_vel_xyz, ang_vel_xyz).
        if linear_velocity is not None or angular_velocity is not None:
            if linear_velocity is not None and angular_velocity is not None:
                if linear_velocity.dim() == 1:
                    linear_velocity = linear_velocity.unsqueeze(0)
                if angular_velocity.dim() == 1:
                    angular_velocity = angular_velocity.unsqueeze(0)
                vel_6d = torch.cat([linear_velocity, angular_velocity], dim=-1)
                entity.set_dofs_velocity(vel_6d, dofs_idx_local=slice(0, 6), envs_idx=env_ids)
            elif linear_velocity is not None:
                entity.set_dofs_velocity(linear_velocity, dofs_idx_local=slice(0, 3), envs_idx=env_ids)
            else:
                entity.set_dofs_velocity(angular_velocity, dofs_idx_local=slice(3, 6), envs_idx=env_ids)
