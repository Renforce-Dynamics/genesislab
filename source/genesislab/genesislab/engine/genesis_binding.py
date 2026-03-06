"""Genesis engine binding layer for GenesisLab.

This module provides a thin binding layer that exposes a stable, RL-centric
interface over a Genesis Scene and entities. It encapsulates engine-specific
details and provides batched state access and control methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import genesis as gs
import torch

if TYPE_CHECKING:
    from genesislab.components.entities.scene_cfg import SceneCfg

from genesislab.engine.assets.articulation import GenesisArticulation, GenesisArticulationCfg


class GenesisBinding:
    """Binding layer for Genesis Scene and entities.

    This class provides a clean interface for RL environments to interact with
    Genesis without directly accessing solver internals. It manages:
    - Scene construction and building
    - Entity loading and indexing
    - State queries (joint states, root poses, velocities)
    - Control target setting
    - Reset operations
    - DOF indexing

    All operations are batched over num_envs environments.
    """

    def __init__(self, cfg: SceneCfg, device: str = "cuda"):
        """Initialize the Genesis binding.

        Args:
            cfg: Scene configuration.
            device: Device to use for tensors ('cuda' or 'cpu').
        """
        self.cfg = cfg
        self.device = device
        self._scene: gs.Scene = None
        self._entities: dict[str, Any] = {}
        self._dof_indices: dict[str, torch.Tensor] = {}
        self._actuators: dict[str, dict[str, Any]] = {}  # entity_name -> {actuator_name -> actuator_instance}
        self._num_envs = cfg.num_envs

    @property
    def scene(self) -> gs.Scene:
        """The Genesis Scene instance."""
        if self._scene is None:
            raise RuntimeError("Scene not built. Call build() first.")
        return self._scene

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def entities(self) -> dict[str, Any]:
        """Dictionary of entity objects keyed by name."""
        return self._entities

    def build(self) -> None:
        """Build the Genesis scene and entities.

        This method:
        1. Creates a Genesis Scene with appropriate options
        2. Adds robots and terrain according to cfg
        3. Builds the scene with num_envs
        4. Resolves DOF indices for controlled joints
        """
        # Create scene with simulation options
        sim_options = gs.options.SimOptions(**self.cfg.to_genesis_options())
        # Viewer options: we respect ``SceneCfg.viewer`` to control whether a
        # window is shown. This keeps the default behaviour (viewer on) but
        # allows scripts to disable it for headless runs.
        viewer_options = gs.options.ViewerOptions()

        self._scene = gs.Scene(
            sim_options=sim_options,
            viewer_options=viewer_options,
            show_viewer=getattr(self.cfg, "viewer", True),
        )

        # Add terrain if specified
        if self.cfg.terrain is not None:
            self._add_terrain()

        # Add robots
        for entity_name, robot_cfg in self.cfg.robots.items():
            self._add_robot(entity_name, robot_cfg)

        # Add sensors if specified
        for sensor_name, sensor_cfg in self.cfg.sensors.items():
            self._add_sensor(sensor_name, sensor_cfg)

        # Optional: attach a simple camera and start video recording before
        # building the scene, if requested via the scene config.
        video_path = getattr(self.cfg, "record_video_path", None)
        if video_path is not None:
            from genesislab.engine.visualize import attach_video_recorder
            attach_video_recorder(self._scene, str(video_path))

        # Build the scene
        self._scene.build(
            n_envs=self.cfg.num_envs,
            env_spacing=self.cfg.env_spacing,
            n_envs_per_row=self.cfg.n_envs_per_row,
            center_envs_at_origin=self.cfg.center_envs_at_origin,
        )

        # Resolve DOF indices for controlled joints
        self._resolve_dof_indices()

        # Process actuator configurations (IsaacLab-style)
        # This takes precedence over legacy PD gains
        self._process_actuators_cfg()

        # Apply default PD gains from robot configs, if specified
        # (only if actuators are not configured)
        self._apply_robot_pd_gains()

    def _process_actuators_cfg(self) -> None:
        """Process and apply actuator configurations for robots (IsaacLab-style).

        This method processes actuator configurations from RobotCfg.actuators and:
        1. Creates actuator instances for each actuator group
        2. For implicit actuators: Sets stiffness/damping to the Genesis engine
        3. For explicit actuators: Sets engine kp/kv to 0 (actuator computes torques)

        If actuators are configured, they take precedence over legacy PD gains.
        """
        from genesislab.components.actuators import ActuatorBase, ImplicitActuator
        from genesislab.utils.configclass.string import resolve_matching_names
        import logging

        logger = logging.getLogger(__name__)

        for entity_name, robot_cfg in self.cfg.robots.items():
            actuators_cfg = getattr(robot_cfg, "actuators", None)
            if actuators_cfg is None:
                continue

            entity = self._entities[entity_name]
            self._actuators[entity_name] = {}

            # Get all joint names from the entity (exclude fixed joints and base)
            # Note: Genesis may not expose joint.type directly, so we get all joints
            # and filter out fixed joints by checking if they have DOFs
            # Also exclude 'base' joint as it's not actuated (floating base)
            joint_names = []
            for joint in entity.joints:
                # Check if joint has DOFs (non-fixed joints have dof_start)
                # Exclude 'base' joint as it's not actuated (floating base)
                if (hasattr(joint, "dof_start") and joint.dof_start is not None 
                    and joint.name.lower() != "base"):
                    joint_names.append(joint.name)
            if not joint_names:
                logger.warning(f"Robot '{entity_name}': No actuated joints found. Skipping actuator processing.")
                continue

            # Get joint state to infer number of DOFs
            dof_pos, _ = self.get_joint_state(entity_name)
            num_dofs = dof_pos.shape[-1]
            num_envs = dof_pos.shape[0]

            # Build joint name to DOF index mapping
            joint_name_to_dof_indices = {}
            for joint in entity.joints:
                # Only process joints that have DOFs
                if not hasattr(joint, "dof_start") or joint.dof_start is None:
                    continue
                joint_name = joint.name
                dof_start = joint.dof_start
                # Genesis joints typically have 1 DOF per joint, but we check for dof_count
                dof_count = getattr(joint, "dof_count", 1) if hasattr(joint, "dof_count") else 1
                joint_name_to_dof_indices[joint_name] = list(range(dof_start, dof_start + dof_count))

            # Process each actuator group
            for actuator_name, actuator_cfg in actuators_cfg.items():
                # Find matching joints using regex
                try:
                    joint_ids, matched_joint_names = resolve_matching_names(
                        actuator_cfg.joint_names_expr, joint_names, preserve_order=False
                    )
                except ValueError as e:
                    logger.error(f"Robot '{entity_name}': Actuator '{actuator_name}': {e}")
                    raise

                if not matched_joint_names:
                    raise ValueError(
                        f"Robot '{entity_name}': Actuator '{actuator_name}': "
                        f"No joints matched expression {actuator_cfg.joint_names_expr}. "
                        f"Available joints: {joint_names}"
                    )

                # Resolve DOF indices for matched joints
                matched_dof_indices = []
                for joint_name in matched_joint_names:
                    matched_dof_indices.extend(joint_name_to_dof_indices[joint_name])
                num_actuator_joints = len(matched_dof_indices)

                # Convert to tensor or slice for efficiency
                if len(matched_joint_names) == len(joint_names):
                    joint_ids_tensor = slice(None)
                else:
                    joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=self.device)

                # Get default joint properties from entity (for now, use zeros as defaults)
                # In a full implementation, these would be read from the USD/URDF file
                default_stiffness = torch.zeros(num_envs, num_actuator_joints, device=self.device)
                default_damping = torch.zeros(num_envs, num_actuator_joints, device=self.device)
                default_armature = torch.zeros(num_envs, num_actuator_joints, device=self.device)
                default_friction = torch.zeros(num_envs, num_actuator_joints, device=self.device)
                default_dynamic_friction = torch.zeros(num_envs, num_actuator_joints, device=self.device)
                default_viscous_friction = torch.zeros(num_envs, num_actuator_joints, device=self.device)
                default_effort_limit = torch.full((num_envs, num_actuator_joints), float('inf'), device=self.device)
                default_velocity_limit = torch.full((num_envs, num_actuator_joints), float('inf'), device=self.device)

                # Create actuator instance
                actuator: ActuatorBase = actuator_cfg.class_type(
                    cfg=actuator_cfg,
                    joint_names=matched_joint_names,
                    joint_ids=joint_ids_tensor,
                    num_envs=num_envs,
                    device=self.device,
                    stiffness=default_stiffness,
                    damping=default_damping,
                    armature=default_armature,
                    friction=default_friction,
                    dynamic_friction=default_dynamic_friction,
                    viscous_friction=default_viscous_friction,
                    effort_limit=default_effort_limit,
                    velocity_limit=default_velocity_limit,
                )

                # Store actuator instance
                self._actuators[entity_name][actuator_name] = actuator

                # Apply actuator configuration to engine
                if isinstance(actuator, ImplicitActuator):
                    # For implicit actuators: set stiffness/damping to engine
                    # Convert actuator stiffness/damping to kp/kv tensors
                    kp = actuator.stiffness[0]  # Shape: (num_joints,)
                    kd = actuator.damping[0]  # Shape: (num_joints,)
                    
                    # Set to engine using DOF indices
                    dof_indices_tensor = torch.tensor(matched_dof_indices, dtype=torch.long, device=self.device)
                    entity.set_dofs_kp(kp, dof_indices_tensor)
                    entity.set_dofs_kv(kd, dof_indices_tensor)
                    
                    logger.info(
                        f"Robot '{entity_name}': Actuator '{actuator_name}' (implicit): "
                        f"Set kp/kv to engine for joints {matched_joint_names}"
                    )
                else:
                    # For explicit actuators: set engine kp/kv to 0
                    # The actuator will compute torques explicitly
                    dof_indices_tensor = torch.tensor(matched_dof_indices, dtype=torch.long, device=self.device)
                    zero_kp = torch.zeros(len(matched_dof_indices), device=self.device)
                    zero_kd = torch.zeros(len(matched_dof_indices), device=self.device)
                    entity.set_dofs_kp(zero_kp, dof_indices_tensor)
                    entity.set_dofs_kv(zero_kd, dof_indices_tensor)
                    
                    logger.info(
                        f"Robot '{entity_name}': Actuator '{actuator_name}' (explicit): "
                        f"Set engine kp/kv to 0 for joints {matched_joint_names}. "
                        f"Actuator will compute torques explicitly."
                    )

    def _apply_robot_pd_gains(self) -> None:
        """Apply PD gains specified in the robot configs (legacy system).

        This method processes PD gains from RobotCfg in the following order:
        1. If ``pd_gains`` (per-joint dict) is specified, it uses those values (regex patterns supported)
        2. Otherwise, if ``default_pd_kp`` and ``default_pd_kd`` are set, it applies uniform gains to all DOFs

        The PD gains are set directly to the Genesis engine via ``set_dofs_kp`` and ``set_dofs_kv``,
        which configures the engine-level PD controller for position control.

        Note:
            This method is only called if actuators are NOT configured. If RobotCfg.actuators is specified,
            the actuator system takes precedence and this method is skipped.

        Note:
            These PD gains are separate from actuator configurations. Actuator models (e.g., IdealPDActuator)
            have their own stiffness/damping parameters that are used for torque computation in explicit
            actuator models. For implicit actuators, the actuator's stiffness/damping are also set to the
            engine, but they are configured through the actuator system, not through RobotCfg.
        """
        import re
        import torch  # Local import to avoid any circular import issues.
        import logging

        logger = logging.getLogger(__name__)

        for entity_name, robot_cfg in self.cfg.robots.items():
            # Skip if actuators are configured (actuators take precedence)
            if getattr(robot_cfg, "actuators", None) is not None:
                logger.debug(
                    f"Robot '{entity_name}': Actuators configured. Skipping legacy PD gain application."
                )
                continue
            entity = self._entities[entity_name]
            
            # Get joint state to infer number of DOFs
            dof_pos, _ = self.get_joint_state(entity_name)
            num_dofs = dof_pos.shape[-1]

            # Initialize gain tensors (will be filled based on config)
            kp_tensor = torch.zeros((num_dofs,), device=self.device)
            kd_tensor = torch.zeros((num_dofs,), device=self.device)
            gains_set = torch.zeros((num_dofs,), dtype=torch.bool, device=self.device)

            # Priority 1: Process per-joint pd_gains if specified
            pd_gains = getattr(robot_cfg, "pd_gains", None)
            if pd_gains is not None:
                # Get all joints from the entity (exclude fixed joints)
                joints = entity.joints
                
                # Build joint name to DOF index mapping
                joint_name_to_dof_indices = {}
                for joint in joints:
                    # Only process joints that have DOFs
                    if not hasattr(joint, "dof_start") or joint.dof_start is None:
                        continue
                    joint_name = joint.name
                    dof_start = joint.dof_start
                    dof_count = getattr(joint, "dof_count", 1) if hasattr(joint, "dof_count") else 1
                    joint_name_to_dof_indices[joint_name] = list(range(dof_start, dof_start + dof_count))
                
                # Apply per-joint gains using regex matching
                for joint_name_pattern, (kp, kd) in pd_gains.items():
                    # Find matching joints using regex
                    pattern = re.compile(joint_name_pattern)
                    matching_joints = [name for name in joint_name_to_dof_indices.keys() if pattern.match(name)]
                    
                    if not matching_joints:
                        import warnings
                        warnings.warn(
                            f"Robot '{entity_name}': No joints matched pattern '{joint_name_pattern}'. "
                            f"Available joints: {list(joint_name_to_dof_indices.keys())}"
                        )
                        continue
                    
                    # Apply gains to all matching joints
                    for joint_name in matching_joints:
                        dof_indices = joint_name_to_dof_indices[joint_name]
                        for dof_idx in dof_indices:
                            if dof_idx < num_dofs:
                                kp_tensor[dof_idx] = float(kp)
                                kd_tensor[dof_idx] = float(kd)
                                gains_set[dof_idx] = True

            # Priority 2: Apply default uniform gains if per-joint gains not set
            if not gains_set.all():
                kp = getattr(robot_cfg, "default_pd_kp", None)
                kd = getattr(robot_cfg, "default_pd_kd", None)
                if kp is not None and kd is not None:
                    # Apply uniform gains to all DOFs that haven't been set
                    mask = ~gains_set
                    kp_tensor[mask] = float(kp)
                    kd_tensor[mask] = float(kd)
                elif kp is not None or kd is not None:
                    # If only one is set, warn and skip
                    import warnings
                    warnings.warn(
                        f"Robot '{entity_name}': Both default_pd_kp and default_pd_kd must be set. "
                        f"Skipping PD gain application."
                    )
                    continue

            # Only apply if we have valid gains
            if (kp_tensor > 0).any() and (kd_tensor > 0).any():
                self.set_pd_gains(entity_name, kp_tensor, kd_tensor)

    def _add_terrain(self) -> None:
        """Add terrain entity to the scene."""
        terrain_cfg = self.cfg.terrain
        if terrain_cfg is None:
            return

        # Support both dict-based and configclass-based terrain configs.
        if isinstance(terrain_cfg, dict):
            terrain_type = terrain_cfg.get("type", "plane")
        else:
            terrain_type = getattr(terrain_cfg, "type", "plane")
        if terrain_type == "plane":
            # Use Genesis' built-in infinite plane primitive so that the ground
            # is visually more obvious in renderings (with proper shading and
            # reflections) while still acting as a flat contact surface.
            plane = gs.morphs.Plane()
            self._scene.add_entity(plane, name="terrain")
        else:
            # Handle other terrain types (e.g., heightfield, mesh)
            raise NotImplementedError(f"Terrain type '{terrain_type}' not yet implemented")

    def _add_robot(self, entity_name: str, robot_cfg: Any) -> None:
        """Add a robot entity to the scene using the Genesis-native asset layer.

        Args:
            entity_name: Name to assign to the entity.
            robot_cfg: Robot configuration.
        """
        if self._scene is None:
            raise RuntimeError("Scene must be created before adding robots.")

        asset_cfg = GenesisArticulationCfg(
            name=entity_name,
            morph_type=robot_cfg.morph_type,
            morph_path=robot_cfg.morph_path,
            initial_pose=robot_cfg.initial_pose,
            fixed_base=robot_cfg.fixed_base,
            control_dofs=robot_cfg.control_dofs,
            morph_options=robot_cfg.morph_options,
        )
        asset = GenesisArticulation(asset_cfg, device=self.device)
        entity = asset.build_into_scene(self._scene)

        # Store only the underlying entity for now; environments can optionally
        # keep track of the wrapper if they need richer behaviour.
        self._entities[entity_name] = entity

    def _add_sensor(self, sensor_name: str, sensor_cfg: dict[str, Any]) -> None:
        """Add a sensor to the scene.

        Args:
            sensor_name: Name to assign to the sensor.
            sensor_cfg: Sensor configuration.
        """
        # Sensor creation logic would go here
        # This is a placeholder for future sensor support
        pass

    def _resolve_dof_indices(self) -> None:
        """Resolve DOF indices for controlled joints in each robot."""
        for entity_name, robot_cfg in self.cfg.robots.items():
            entity = self._entities[entity_name]
            control_dofs = robot_cfg.control_dofs

            if control_dofs is None:
                # If no specific joints specified, get all actuated joints
                # This would require querying the entity for joint names
                # For now, we'll store None and resolve later when needed
                self._dof_indices[entity_name] = None
            else:
                # Resolve joint names to DOF indices
                dof_indices = []
                for joint_name in control_dofs:
                    joint = entity.get_joint(joint_name)
                    if joint is not None:
                        # Genesis entities expose dof_start for each joint
                        dof_start = joint.dof_start
                        dof_count = joint.dof_count if hasattr(joint, "dof_count") else 1
                        dof_indices.extend(range(dof_start, dof_start + dof_count))
                    else:
                        raise ValueError(f"Joint '{joint_name}' not found in entity '{entity_name}'")

                self._dof_indices[entity_name] = torch.tensor(dof_indices, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor = None) -> None:
        """Reset specified environments to initial state.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            self.scene.reset()
        else:
            # Convert to list if tensor
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().tolist()
            self.scene.reset(envs_idx=env_ids)

    def step(self) -> None:
        """Step the physics simulation by one timestep."""
        self.scene.step()

    def get_joint_state(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint positions and velocities for an entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            Tuple of (positions, velocities) tensors of shape (num_envs, num_dofs).
        """
        entity = self._entities[entity_name]
        dof_indices = self._dof_indices.get(entity_name)

        # Get joint positions
        if dof_indices is not None:
            positions = entity.get_dofs_position(dof_indices)
        else:
            # Get all DOF positions
            positions = entity.get_dofs_position()

        # Get joint velocities
        if dof_indices is not None:
            velocities = entity.get_dofs_velocity(dof_indices)
        else:
            # Get all DOF velocities
            velocities = entity.get_dofs_velocity()

        return positions, velocities

    def get_root_state(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get root pose and velocity for an entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            Tuple of (position, quaternion, linear_velocity, angular_velocity) tensors.
            Positions and velocities have shape (num_envs, 3).
            Quaternions have shape (num_envs, 4).
        """
        entity = self._entities[entity_name]

        # Get root pose
        position = entity.get_pos()
        quaternion = entity.get_quat()

        # Get root velocity
        linear_velocity = entity.get_vel()
        angular_velocity = entity.get_ang()

        return position, quaternion, linear_velocity, angular_velocity

    def get_body_positions(self, entity_name: str) -> torch.Tensor:
        """Get positions of all bodies/links for an entity.

        This uses the Genesis ``RigidEntity`` API:

        - Prefer the vectorized ``get_links_pos()`` method, which returns all link
          positions in one call.
        - Fall back to iterating over individual links via ``get_link(i).get_pos()``
          if the batched API is not available.

        Args:
            entity_name: Name of the entity.

        Returns:
            Tensor of shape ``(num_envs, num_links, 3)`` containing the world-frame
            positions of all links/bodies.
        """
        entity = self._entities[entity_name]

        # Fast path: Genesis >= 0.3.x exposes a batched API.
        if hasattr(entity, "get_links_pos"):
            return entity.get_links_pos()

        # Fallback: iterate over links if we can query them individually.
        # RigidEntity usually exposes ``n_links`` and ``get_link(idx)``.
        if hasattr(entity, "n_links") and hasattr(entity, "get_link"):
            num_links = int(entity.n_links)
            # Query first link to infer (num_envs, 3) shape and device
            if num_links == 0:
                return torch.zeros((self.num_envs, 0, 3), device=self.device)

            first_link = entity.get_link(0)
            if not hasattr(first_link, "get_pos"):
                raise AttributeError(
                    f"Link 0 of entity '{entity_name}' does not have a 'get_pos' method."
                )
            first_pos = first_link.get_pos()  # (num_envs, 3)
            num_envs = first_pos.shape[0]
            positions = torch.empty(
                (num_envs, num_links, 3),
                device=first_pos.device,
                dtype=first_pos.dtype,
            )
            positions[:, 0, :] = first_pos

            for i in range(1, num_links):
                link = entity.get_link(i)
                if not hasattr(link, "get_pos"):
                    raise AttributeError(
                        f"Link {i} of entity '{entity_name}' does not have a 'get_pos' method."
                    )
                positions[:, i, :] = link.get_pos()

            return positions

        # If we reach here, the underlying Genesis entity API is not what we expect.
        raise AttributeError(
            f"Entity '{entity_name}' does not expose 'get_links_pos' or "
            f"('n_links' and 'get_link') methods; cannot fetch link positions."
        )

    def set_joint_targets(self, entity_name: str, targets: torch.Tensor, control_type: str = "position") -> None:
        """Set joint control targets for an entity.

        Args:
            entity_name: Name of the entity.
            targets: Target values of shape (num_envs, num_dofs).
            control_type: Type of control ('position', 'velocity', or 'torque').
        """
        entity = self._entities[entity_name]
        dof_indices = self._dof_indices.get(entity_name)

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
        entity = self._entities[entity_name]
        dof_indices = self._dof_indices.get(entity_name)

        if dof_indices is not None:
            entity.set_dofs_kp(kp, dof_indices)
            entity.set_dofs_kv(kd, dof_indices)
        else:
            entity.set_dofs_kp(kp)
            entity.set_dofs_kv(kd)
