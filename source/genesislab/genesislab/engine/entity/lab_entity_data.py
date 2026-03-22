from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from genesislab.envs.manager_based_genesis_env import ManagerBasedGenesisEnv
    from .lab_entity import LabEntity


class LabEntityData:
    """Data container for an entity in the simulation.

    This class provides lazy-loaded access to entity state data, similar to
    IsaacLab's ArticulationData. All data is fetched on-demand from the
    underlying scene layer.

    The data includes:
    - Joint state: positions and velocities
    - Root state: position, quaternion, linear and angular velocities
    - Link positions: world frame positions of all links/bodies
    """

    def __init__(self, env: "ManagerBasedGenesisEnv", entity: "LabEntity"):
        """Initialize the entity data view.

        Args:
            env: The environment instance.
            entity: LabEntity wrapper for the underlying Genesis entity.
        """
        self._env = env
        self._scene = env.scene
        self._lab_entity = entity
        self._entity_name = entity.name
        self._raw_entity = entity.raw_entity
        # Track previous joint velocity for acceleration computation
        self._prev_joint_vel: torch.Tensor = None
        # Track last step when acceleration was computed (to avoid multiple updates per step)
        self._last_acc_step: int = -1

    _default_joint_pos: torch.Tensor = None
    _default_joint_vel: torch.Tensor = None
    _default_root_pos: torch.Tensor = None
    _default_root_quat: torch.Tensor = None
    _default_root_lin_vel: torch.Tensor = None
    _default_root_ang_vel: torch.Tensor = None
    _soft_joint_pos_limits: torch.Tensor = None

    # ------------------------------------------------------------------
    # Name accessors
    # ------------------------------------------------------------------

    @property
    def joint_names(self) -> list[str]:
        """Normalized actuated joint names (floating base excluded)."""
        return self._lab_entity.joint_names

    @property
    def link_names(self) -> list[str]:
        """Link/body names (all)."""
        return self._lab_entity.link_names

    @property
    def raw_link_names(self) -> list[str]:
        """Raw link/body names (all)."""
        return self._lab_entity.raw_link_names

    @property
    def GRAVITY_VEC_W(self) -> torch.Tensor:
        """World-frame gravity vector (0, 0, -1), shape (num_envs, 3). Used e.g. for orientation error in tracking."""
        return torch.tensor([0.0, 0.0, -1.0], device=self._env.device, dtype=torch.float32).unsqueeze(0).expand(
            self._env.num_envs, 3
        )

    @property
    def default_joint_pos(self) -> torch.Tensor:
        """Default joint positions. Shape: (num_envs, num_dofs).

        This quantity is configured through the robot configuration's `default_joint_pos` parameter.
        If not configured, returns zeros.
        """
        if self._default_joint_pos is None:
            # Initialize default joint positions
            # Get current joint positions to infer full DOF shape, then drop base DOFs
            joint_pos_full = self._raw_entity.get_dofs_position()
            num_envs, num_dofs_full = joint_pos_full.shape
            base_offset = 6 if num_dofs_full > 6 else 0
            num_joints = num_dofs_full - base_offset

            # Initialize with zeros for joint DOFs only (exclude base)
            self._default_joint_pos = torch.zeros(num_envs, num_joints, device=self._env.device)
            
            robot_cfg = self._env.scene.cfg.robots.get(self._entity_name)
            if robot_cfg is not None and hasattr(robot_cfg, "default_joint_pos") and robot_cfg.default_joint_pos is not None:
                # Get Robot asset from Entity (required)
                entity = self._env.entities[self._entity_name]
                robot_asset = entity.robot_asset
                if robot_asset is None:
                    raise RuntimeError(
                        f"Robot asset for entity '{self._entity_name}' is None. "
                        f"This indicates that the robot was not properly initialized with GenesisArticulationRobot."
                    )
                
                # Use Robot's name resolution to resolve joint values and DOF indices
                joint_values = robot_asset.resolve_joint_values(robot_cfg.default_joint_pos)
                joint_dof_indices = robot_asset.get_all_joint_dof_indices()
                
                # Set default joint positions for matched joints (map global DOF -> joint-subspace index)
                for joint_name, value in joint_values.items():
                    if joint_name in joint_dof_indices:
                        dof_indices_full = joint_dof_indices[joint_name]
                        for dof_idx_full in dof_indices_full:
                            if base_offset <= dof_idx_full < num_dofs_full:
                                dof_idx_joint = dof_idx_full - base_offset
                                if dof_idx_joint < num_joints:
                                    self._default_joint_pos[:, dof_idx_joint] = value

        return self._default_joint_pos

    @property
    def default_joint_vel(self) -> torch.Tensor:
        """Default joint velocities. Shape: (num_envs, num_dofs).

        This quantity is configured through the robot configuration's `default_joint_vel` parameter.
        If not configured, returns zeros (default velocity is zero).
        """
        if self._default_joint_vel is None:
            # Initialize default joint velocities
            # Get current joint velocities to infer full DOF shape, then drop base DOFs
            joint_vel_full = self._raw_entity.get_dofs_velocity()
            num_envs, num_dofs_full = joint_vel_full.shape
            base_offset = 6 if num_dofs_full > 6 else 0
            num_joints = num_dofs_full - base_offset

            # Initialize with zeros (default velocity is zero) for joint DOFs only
            self._default_joint_vel = torch.zeros(num_envs, num_joints, device=self._env.device)
            
            robot_cfg = self._env.scene.cfg.robots.get(self._entity_name)
            if robot_cfg is not None and hasattr(robot_cfg, "default_joint_vel") and robot_cfg.default_joint_vel is not None:
                # Get Robot asset from Entity (required)
                entity = self._env.entities[self._entity_name]
                robot_asset = entity.robot_asset
                if robot_asset is None:
                    raise RuntimeError(
                        f"Robot asset for entity '{self._entity_name}' is None. "
                        f"This indicates that the robot was not properly initialized with GenesisArticulationRobot."
                    )
                
                # Use Robot's name resolution to resolve joint values and DOF indices
                joint_values = robot_asset.resolve_joint_values(robot_cfg.default_joint_vel)
                joint_dof_indices = robot_asset.get_all_joint_dof_indices()
                
                # Set default joint velocities for matched joints (map global DOF -> joint-subspace index)
                for joint_name, value in joint_values.items():
                    if joint_name in joint_dof_indices:
                        dof_indices_full = joint_dof_indices[joint_name]
                        for dof_idx_full in dof_indices_full:
                            if base_offset <= dof_idx_full < num_dofs_full:
                                dof_idx_joint = dof_idx_full - base_offset
                                if dof_idx_joint < num_joints:
                                    self._default_joint_vel[:, dof_idx_joint] = value

        return self._default_joint_vel

    @property
    def joint_pos(self) -> torch.Tensor:
        """Joint positions. Shape: (num_envs, num_dofs)."""
        pos_full = self._raw_entity.get_dofs_position()
        base_offset = 6
        return pos_full[:, base_offset:]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Joint velocities. Shape: (num_envs, num_dofs)."""
        vel_full = self._raw_entity.get_dofs_velocity()
        base_offset = 6
        return vel_full[:, base_offset:]

    @property
    def joint_acc(self) -> torch.Tensor:
        """Joint accelerations. Shape: (num_envs, num_dofs).
        
        This property computes joint accelerations by numerical differentiation
        of joint velocities. The acceleration is computed as:
        acc = (vel_current - vel_previous) / dt
        
        On the first call or after reset, returns zeros (no previous velocity available).
        The previous velocity is updated once per environment step to ensure consistency.
        """
        # Get current joint velocity (joint DOFs only, exclude base)
        vel_full = self._raw_entity.get_dofs_velocity()
        num_envs, num_dofs_full = vel_full.shape
        base_offset = 6
        vel_current = vel_full[:, base_offset:]
        num_dofs = vel_current.shape[-1]
        
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
    def soft_joint_pos_limits(self) -> torch.Tensor:
        """Soft joint position limits. Shape: (num_envs, num_dofs, 2).

        Limits are taken directly from Genesis DOF limits (per-DOF lower/upper)
        and mapped into the joint subspace (excluding base DOFs).
        """
        if self._soft_joint_pos_limits is None:
            # Genesis returns a tuple (lower, upper), each of shape (n_dofs,)
            lower, upper = self._raw_entity.get_dofs_limit()
            # Move to env device if needed
            lower = lower.to(self._env.device)
            upper = upper.to(self._env.device)

            n_dofs_full = lower.shape[0]
            base_offset = 6 if n_dofs_full > 6 else 0
            # Joint DOFs only (exclude base)
            lower_joint = lower[base_offset:]
            upper_joint = upper[base_offset:]

            # Stack into (num_joints, 2) as [min, max]
            limits_joint = torch.stack([lower_joint, upper_joint], dim=-1)  # (num_joints, 2)
            num_envs = self._env.num_envs

            # Expand to (num_envs, num_joints, 2)
            self._soft_joint_pos_limits = limits_joint.unsqueeze(0).expand(num_envs, -1, -1)

        return self._soft_joint_pos_limits

    @property
    def applied_torque(self) -> torch.Tensor:
        """Applied joint torques/efforts. Shape: (num_envs, num_dofs).
        
        This property collects applied efforts from all actuators configured for this entity.
        If no actuators are configured, returns zeros.
        """
        # Get joint state to infer shape (joint DOFs only, exclude base)
        joint_pos_full = self._raw_entity.get_dofs_position()
        num_envs, num_dofs_full = joint_pos_full.shape
        base_offset = 6 if num_dofs_full > 6 else 0
        num_dofs = num_dofs_full - base_offset
        
        # Initialize with zeros
        applied_torques = torch.zeros(num_envs, num_dofs, device=self._env.device)
        
        entity = self._env.entities[self._entity_name]
        entity_actuators = entity.actuators
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
        return self._raw_entity.get_pos()

    @property
    def link_pos_w(self) -> torch.Tensor:
        """All link positions in world frame. Shape: (num_envs, num_links, 3).
        
        Returns the translation (position) of all links/bodies in the entity.
        """
        return self._raw_entity.get_links_pos()

    # IsaacLab-style body accessors used by imitation/tracking MDP
    @property
    def body_pos_w(self) -> torch.Tensor:
        """Alias for link positions in world frame. Shape: (num_envs, num_links, 3)."""
        return self.link_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """All link orientations in world frame. Shape: (num_envs, num_links, 4).

        Requires that the underlying Genesis entity exposes per-link quaternions.
        """
        if not hasattr(self._raw_entity, "get_links_quat"):
            raise RuntimeError(
                f"Raw entity for '{self._entity_name}' does not expose 'get_links_quat'. "
                "Per-link orientations are required for imitation/tracking MDP."
            )
        return self._raw_entity.get_links_quat()  # type: ignore[attr-defined]

    @property
    def body_rot_w(self) -> torch.Tensor:
        """All link orientations in world frame. Shape: (num_envs, num_links, 4).

        Requires that the underlying Genesis entity exposes per-link quaternions.
        """
        return self._raw_entity.get_dofs_position()[:, 3:6]  # type: ignore[attr-defined]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """All link linear velocities in world frame. Shape: (num_envs, num_links, 3).

        Requires that the underlying Genesis entity exposes per-link linear velocities.
        """
        if not hasattr(self._raw_entity, "get_links_vel"):
            raise RuntimeError(
                f"Raw entity for '{self._entity_name}' does not expose 'get_links_vel'. "
                "Per-link linear velocities are required for imitation/tracking MDP."
            )
        return self._raw_entity.get_links_vel()  # type: ignore[attr-defined]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """All link angular velocities in world frame. Shape: (num_envs, num_links, 3).

        Requires that the underlying Genesis entity exposes per-link angular velocities.
        """
        if not hasattr(self._raw_entity, "get_links_ang"):
            raise RuntimeError(
                f"Raw entity for '{self._entity_name}' does not expose 'get_links_ang'. "
                "Per-link angular velocities are required for imitation/tracking MDP."
            )
        return self._raw_entity.get_links_ang()  # type: ignore[attr-defined]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root quaternion in world frame. Shape: (num_envs, 4)."""
        return self._raw_entity.get_quat()

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in world frame. Shape: (num_envs, 3)."""
        return self._raw_entity.get_vel()

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in world frame. Shape: (num_envs, 3)."""
        return self._raw_entity.get_ang()

    # ------------------------------------------------------------------
    # Default root state (capture once, used as reset reference)
    # ------------------------------------------------------------------

    @property
    def default_root_pos_w(self) -> torch.Tensor:
        """Default root position used for resets. Shape: (num_envs, 3)."""
        if self._default_root_pos is None:
            self._default_root_pos = self.root_pos_w.clone()
        return self._default_root_pos

    @property
    def default_root_quat_w(self) -> torch.Tensor:
        """Default root orientation used for resets. Shape: (num_envs, 4)."""
        if self._default_root_quat is None:
            self._default_root_quat = self.root_quat_w.clone()
        return self._default_root_quat

    @property
    def default_root_lin_vel_w(self) -> torch.Tensor:
        """Default root linear velocity used for resets. Shape: (num_envs, 3)."""
        if self._default_root_lin_vel is None:
            # Typically zero, but capture whatever the initial state is.
            self._default_root_lin_vel = self.root_lin_vel_w.clone()
        return self._default_root_lin_vel

    @property
    def default_root_ang_vel_w(self) -> torch.Tensor:
        """Default root angular velocity used for resets. Shape: (num_envs, 3)."""
        if self._default_root_ang_vel is None:
            self._default_root_ang_vel = self.root_ang_vel_w.clone()
        return self._default_root_ang_vel

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