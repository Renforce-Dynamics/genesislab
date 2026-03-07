"""Actuator and PD gains management for GenesisBinding."""

from __future__ import annotations

from typing import Any

import torch


class ActuatorManager:
    """Helper class for managing actuators and PD gains."""

    def __init__(self, binding: Any):
        """Initialize the actuator manager.

        Args:
            binding: Reference to the GenesisBinding instance.
        """
        self._binding = binding

    def process_actuators_cfg(self) -> None:
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

        for entity_name, robot_cfg in self._binding.cfg.robots.items():
            actuators_cfg = getattr(robot_cfg, "actuators", None)
            if actuators_cfg is None:
                continue

            entity = self._binding._entities[entity_name]
            self._binding._actuators[entity_name] = {}

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
            dof_pos, _ = self._binding.get_joint_state(entity_name)
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
                    joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=self._binding.device)

                # Get default joint properties from entity (for now, use zeros as defaults)
                # In a full implementation, these would be read from the USD/URDF file
                default_stiffness = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_damping = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_armature = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_friction = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_dynamic_friction = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_viscous_friction = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_effort_limit = torch.full((num_envs, num_actuator_joints), float('inf'), device=self._binding.device)
                default_velocity_limit = torch.full((num_envs, num_actuator_joints), float('inf'), device=self._binding.device)

                # Create actuator instance
                actuator: ActuatorBase = actuator_cfg.class_type(
                    cfg=actuator_cfg,
                    joint_names=matched_joint_names,
                    joint_ids=joint_ids_tensor,
                    num_envs=num_envs,
                    device=self._binding.device,
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
                self._binding._actuators[entity_name][actuator_name] = actuator

                # Apply actuator configuration to engine
                if isinstance(actuator, ImplicitActuator):
                    # For implicit actuators: set stiffness/damping to engine
                    # Convert actuator stiffness/damping to kp/kv tensors
                    kp = actuator.stiffness[0]  # Shape: (num_joints,)
                    kd = actuator.damping[0]  # Shape: (num_joints,)
                    
                    # Set to engine using DOF indices
                    dof_indices_tensor = torch.tensor(matched_dof_indices, dtype=torch.long, device=self._binding.device)
                    entity.set_dofs_kp(kp, dof_indices_tensor)
                    entity.set_dofs_kv(kd, dof_indices_tensor)
                    
                    logger.info(
                        f"Robot '{entity_name}': Actuator '{actuator_name}' (implicit): "
                        f"Set kp/kv to engine for joints {matched_joint_names}"
                    )
                else:
                    # For explicit actuators: set engine kp/kv to 0
                    # The actuator will compute torques explicitly
                    dof_indices_tensor = torch.tensor(matched_dof_indices, dtype=torch.long, device=self._binding.device)
                    zero_kp = torch.zeros(len(matched_dof_indices), device=self._binding.device)
                    zero_kd = torch.zeros(len(matched_dof_indices), device=self._binding.device)
                    entity.set_dofs_kp(zero_kp, dof_indices_tensor)
                    entity.set_dofs_kv(zero_kd, dof_indices_tensor)
                    
                    logger.info(
                        f"Robot '{entity_name}': Actuator '{actuator_name}' (explicit): "
                        f"Set engine kp/kv to 0 for joints {matched_joint_names}. "
                        f"Actuator will compute torques explicitly."
                    )

    def apply_robot_pd_gains(self) -> None:
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
        import logging

        logger = logging.getLogger(__name__)

        for entity_name, robot_cfg in self._binding.cfg.robots.items():
            # Skip if actuators are configured (actuators take precedence)
            if getattr(robot_cfg, "actuators", None) is not None:
                logger.debug(
                    f"Robot '{entity_name}': Actuators configured. Skipping legacy PD gain application."
                )
                continue
            entity = self._binding._entities[entity_name]
            
            # Get joint state to infer number of DOFs
            dof_pos, _ = self._binding.get_joint_state(entity_name)
            num_dofs = dof_pos.shape[-1]

            # Initialize gain tensors (will be filled based on config)
            kp_tensor = torch.zeros((num_dofs,), device=self._binding.device)
            kd_tensor = torch.zeros((num_dofs,), device=self._binding.device)
            gains_set = torch.zeros((num_dofs,), dtype=torch.bool, device=self._binding.device)

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
                self._binding.set_pd_gains(entity_name, kp_tensor, kd_tensor)
