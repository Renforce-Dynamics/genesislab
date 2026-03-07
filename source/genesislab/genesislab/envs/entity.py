"""Entity data view for GenesisLab environments.

This module provides a data view abstraction similar to IsaacLab's ArticulationData,
allowing MDP code to access entity state through a clean, typed interface like
`env.entities["go2"].data.joint_pos` instead of calling `env.get_joint_state("go2")`.
"""

from __future__ import annotations

from ast import Raise
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from genesislab.envs.manager_based_genesis_env import ManagerBasedGenesisEnv


class EntityData:
    """Data container for an entity in the simulation.

    This class provides lazy-loaded access to entity state data, similar to
    IsaacLab's ArticulationData. All data is fetched on-demand from the
    underlying binding layer.

    The data includes:
    - Joint state: positions and velocities
    - Root state: position, quaternion, linear and angular velocities
    - Link positions: world frame positions of all links/bodies
    """

    def __init__(self, env: "ManagerBasedGenesisEnv", entity_name: str):
        """Initialize the entity data view.

        Args:
            env: The environment instance.
            entity_name: Name of the entity.
        """
        self._env = env
        self._entity_name = entity_name
        # Track previous joint velocity for acceleration computation
        self._prev_joint_vel: torch.Tensor = None
        # Track last step when acceleration was computed (to avoid multiple updates per step)
        self._last_acc_step: int = -1

    _default_joint_pos: torch.Tensor = None
    _default_joint_vel: torch.Tensor = None

    @property
    def default_joint_pos(self) -> torch.Tensor:
        """Default joint positions. Shape: (num_envs, num_dofs).

        This quantity is configured through the robot configuration's `default_joint_pos` parameter.
        If not configured, returns zeros.
        """
        if self._default_joint_pos is None:
            # Initialize default joint positions
            # Get current joint positions to infer shape
            joint_pos, _ = self._env.get_joint_state(self._entity_name)
            num_envs, num_dofs = joint_pos.shape
            
            # Initialize with zeros
            self._default_joint_pos = torch.zeros(num_envs, num_dofs, device=self._env.device)
            
            robot_cfg = self._env._binding.cfg.robots.get(self._entity_name)
            if robot_cfg is not None and hasattr(robot_cfg, "default_joint_pos") and robot_cfg.default_joint_pos is not None:
                # Get joint names from entity
                entity = self._env._binding.entities[self._entity_name]
                joint_names = []
                joint_name_to_dof_indices = {}
                
                for joint in entity.joints:
                    # Only process joints that have DOFs
                    if hasattr(joint, "dof_start") and joint.dof_start is not None:
                        joint_name = joint.name
                        joint_names.append(joint_name)
                        dof_start = joint.dof_start
                        dof_count = getattr(joint, "dof_count", 1)
                        joint_name_to_dof_indices[joint_name] = list(range(dof_start, dof_start + dof_count))
                
                if joint_names:
                    from genesislab.utils.configclass.string import resolve_matching_names_values
                    indices_list, _, values_list = resolve_matching_names_values(
                        robot_cfg.default_joint_pos, joint_names
                    )
                    # Set default joint positions for matched joints
                    for idx, value in zip(indices_list, values_list):
                        joint_name = joint_names[idx]
                        dof_indices = joint_name_to_dof_indices[joint_name]
                        # Set value for all DOFs of this joint
                        for dof_idx in dof_indices:
                            if dof_idx < num_dofs:
                                self._default_joint_pos[:, dof_idx] = value

        return self._default_joint_pos

    @property
    def default_joint_vel(self) -> torch.Tensor:
        """Default joint velocities. Shape: (num_envs, num_dofs).

        This quantity is configured through the robot configuration's `default_joint_vel` parameter.
        If not configured, returns zeros (default velocity is zero).
        """
        if self._default_joint_vel is None:
            # Initialize default joint velocities
            # Get current joint velocities to infer shape
            _, joint_vel = self._env.get_joint_state(self._entity_name)
            num_envs, num_dofs = joint_vel.shape
            
            # Initialize with zeros (default velocity is zero)
            self._default_joint_vel = torch.zeros(num_envs, num_dofs, device=self._env.device)
            
            robot_cfg = self._env._binding.cfg.robots.get(self._entity_name)
            if robot_cfg is not None and hasattr(robot_cfg, "default_joint_vel") and robot_cfg.default_joint_vel is not None:
                # Get joint names from entity
                entity = self._env._binding.entities[self._entity_name]
                joint_names = []
                joint_name_to_dof_indices = {}
                
                for joint in entity.joints:
                    # Only process joints that have DOFs
                    if hasattr(joint, "dof_start") and joint.dof_start is not None:
                        joint_name = joint.name
                        joint_names.append(joint_name)
                        dof_start = joint.dof_start
                        dof_count = getattr(joint, "dof_count", 1)
                        joint_name_to_dof_indices[joint_name] = list(range(dof_start, dof_start + dof_count))
                
                if joint_names:
                    from genesislab.utils.configclass.string import resolve_matching_names_values
                    indices_list, _, values_list = resolve_matching_names_values(
                        robot_cfg.default_joint_vel, joint_names
                    )
                    # Set default joint velocities for matched joints
                    for idx, value in zip(indices_list, values_list):
                        joint_name = joint_names[idx]
                        dof_indices = joint_name_to_dof_indices[joint_name]
                        # Set value for all DOFs of this joint
                        for dof_idx in dof_indices:
                            if dof_idx < num_dofs:
                                self._default_joint_vel[:, dof_idx] = value

        return self._default_joint_vel

    @property
    def joint_pos(self) -> torch.Tensor:
        """Joint positions. Shape: (num_envs, num_dofs)."""
        pos, _ = self._env.get_joint_state(self._entity_name)
        return pos

    @property
    def joint_vel(self) -> torch.Tensor:
        """Joint velocities. Shape: (num_envs, num_dofs)."""
        _, vel = self._env.get_joint_state(self._entity_name)
        return vel

    @property
    def joint_acc(self) -> torch.Tensor:
        """Joint accelerations. Shape: (num_envs, num_dofs).
        
        This property computes joint accelerations by numerical differentiation
        of joint velocities. The acceleration is computed as:
        acc = (vel_current - vel_previous) / dt
        
        On the first call or after reset, returns zeros (no previous velocity available).
        The previous velocity is updated once per environment step to ensure consistency.
        """
        # Get current joint velocity
        _, vel_current = self._env.get_joint_state(self._entity_name)
        num_envs, num_dofs = vel_current.shape
        
        # Get current step count to track when to update
        current_step = getattr(self._env, "common_step_counter", 0)
        
        # Initialize previous velocity buffer if needed
        if self._prev_joint_vel is None:
            self._prev_joint_vel = vel_current.clone()
            self._last_acc_step = current_step
            # Return zeros on first call (no previous velocity to differentiate)
            return torch.zeros(num_envs, num_dofs, device=self._env.device)
        
        # Check if shape changed (e.g., after reset)
        if self._prev_joint_vel.shape != vel_current.shape:
            self._prev_joint_vel = vel_current.clone()
            self._last_acc_step = current_step
            return torch.zeros(num_envs, num_dofs, device=self._env.device)
        
        # Compute acceleration: (vel_current - vel_previous) / dt
        # Use physics_dt for differentiation
        dt = self._env.physics_dt
        joint_acc = (vel_current - self._prev_joint_vel) / dt
        
        # Update previous velocity only once per step (if step changed)
        if current_step != self._last_acc_step:
            self._prev_joint_vel = vel_current.clone()
            self._last_acc_step = current_step
        
        return joint_acc

    @property
    def applied_torque(self) -> torch.Tensor:
        """Applied joint torques/efforts. Shape: (num_envs, num_dofs).
        
        This property collects applied efforts from all actuators configured for this entity.
        If no actuators are configured, returns zeros.
        """
        # Get joint state to infer shape
        joint_pos, _ = self._env.get_joint_state(self._entity_name)
        num_envs, num_dofs = joint_pos.shape
        
        # Initialize with zeros
        applied_torques = torch.zeros(num_envs, num_dofs, device=self._env.device)
        
        entity_actuators = self._env._binding._actuators.get(self._entity_name, {})
        if not entity_actuators: raise ValueError("The actuators not specified.")
        
        # Collect applied efforts from all actuators
        # Each actuator has applied_effort and joint_ids that map to DOF indices
        for actuator_name, actuator in entity_actuators.items():
            if not hasattr(actuator, "applied_effort"):
                continue
            
            applied_effort = actuator.applied_effort  # (num_envs, num_actuator_dofs)
            
            # Get joint_ids for this actuator (these are DOF indices)
            if not hasattr(actuator, "joint_ids"):
                continue
            
            joint_ids = actuator.joint_ids
            
            # Handle different joint_ids types
            if isinstance(joint_ids, slice):
                # Convert slice to indices
                if joint_ids == slice(None):
                    # All DOFs
                    dof_indices = torch.arange(num_dofs, device=self._env.device)
                else:
                    start = joint_ids.start if joint_ids.start is not None else 0
                    stop = joint_ids.stop if joint_ids.stop is not None else num_dofs
                    step = joint_ids.step if joint_ids.step is not None else 1
                    dof_indices = torch.arange(start, stop, step, device=self._env.device)
            elif isinstance(joint_ids, (list, tuple)):
                dof_indices = torch.tensor(joint_ids, dtype=torch.long, device=self._env.device)
            elif isinstance(joint_ids, torch.Tensor):
                dof_indices = joint_ids.to(device=self._env.device)
            else:
                continue
            
            # Ensure dof_indices are within bounds
            dof_indices = dof_indices[dof_indices < num_dofs]
            if len(dof_indices) == 0:
                continue
            
            # Map applied_effort to the correct DOF indices
            num_actuator_dofs = applied_effort.shape[1]
            num_mapped_dofs = min(len(dof_indices), num_actuator_dofs)
            applied_torques[:, dof_indices[:num_mapped_dofs]] = applied_effort[:, :num_mapped_dofs]
        
        return applied_torques

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in world frame. Shape: (num_envs, 3)."""
        pos, _, _, _ = self._env.get_root_state(self._entity_name)
        return pos

    @property
    def link_pos_w(self) -> torch.Tensor:
        """All link positions in world frame. Shape: (num_envs, num_links, 3).
        
        Returns the translation (position) of all links/bodies in the entity.
        """
        return self._env.get_body_positions(self._entity_name)

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root quaternion in world frame. Shape: (num_envs, 4)."""
        _, quat, _, _ = self._env.get_root_state(self._entity_name)
        return quat

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in world frame. Shape: (num_envs, 3)."""
        _, _, lin_vel, _ = self._env.get_root_state(self._entity_name)
        return lin_vel

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in world frame. Shape: (num_envs, 3)."""
        _, _, _, ang_vel = self._env.get_root_state(self._entity_name)
        return ang_vel

    # For compatibility with IsaacLab-style observations that use body frame
    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in body frame. Shape: (num_envs, 3).

        Note: Currently returns world frame velocity. Body frame transformation
        can be added if needed.
        """
        # For now, return world frame. Can be transformed to body frame if needed.
        return self.root_lin_vel_w

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in body frame. Shape: (num_envs, 3).

        Note: Currently returns world frame velocity. Body frame transformation
        can be added if needed.
        """
        # For now, return world frame. Can be transformed to body frame if needed.
        return self.root_ang_vel_w

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Projected gravity vector in body frame. Shape: (num_envs, 3).

        This computes the gravity vector projected onto the entity's root frame.
        Uses inverse quaternion rotation to transform gravity from world to body frame.
        """
        # Get gravity direction (assumed to be -Z in world frame)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self._env.device).expand(
            self._env.num_envs, 3
        )

        # Get root quaternion
        quat = self.root_quat_w

        # Transform gravity to body frame using inverse quaternion rotation
        # quat format: [x, y, z, w] (Genesis format) - convert to [w, x, y, z] for rotation
        if quat.shape[-1] != 4:
            raise ValueError(
                f"Quaternion must have 4 components, got shape {quat.shape}. "
                f"Expected quaternion format: [x, y, z, w] or [w, x, y, z]."
            )
        
        # Normalize quaternion
        quat_norm = quat / torch.norm(quat, dim=-1, keepdim=True)
        # Extract components (assuming [x, y, z, w] format from Genesis)
        qx, qy, qz, qw = quat_norm[..., 0], quat_norm[..., 1], quat_norm[..., 2], quat_norm[..., 3]

        # Convert to [w, x, y, z] format for rotation (IsaacLab format)
        quat_wxyz = torch.stack([qw, qx, qy, qz], dim=-1)  # (num_envs, 4)

        # Apply inverse quaternion rotation using IsaacLab's formula
        # quat_apply_inverse: v' = v - 2*w*cross(xyz, v) + 2*cross(xyz, cross(xyz, v))
        xyz = quat_wxyz[:, 1:]  # (num_envs, 3)
        w = quat_wxyz[:, 0:1]  # (num_envs, 1)
        t = xyz.cross(gravity_w, dim=-1) * 2  # (num_envs, 3)
        gravity_b = gravity_w - w * t + xyz.cross(t, dim=-1)

        return gravity_b


class Entity:
    """Wrapper for a Genesis entity with data view access.

    This class wraps the underlying Genesis entity and provides a clean
    interface for accessing entity state through the `data` property,
    similar to IsaacLab's Articulation class.
    """

    def __init__(self, env: "ManagerBasedGenesisEnv", entity_name: str, raw_entity: Any):
        """Initialize the entity wrapper.

        Args:
            env: The environment instance.
            entity_name: Name of the entity.
            raw_entity: The underlying Genesis entity object.
        """
        self._env = env
        self._entity_name = entity_name
        self._raw_entity = raw_entity
        self._data = EntityData(env, entity_name)

    @property
    def name(self) -> str:
        """Name of the entity."""
        return self._entity_name

    @property
    def data(self) -> EntityData:
        """Data view for accessing entity state."""
        return self._data

    @property
    def raw_entity(self) -> Any:
        """Access to the underlying Genesis entity (for advanced use cases)."""
        return self._raw_entity
