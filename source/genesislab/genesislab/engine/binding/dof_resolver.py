"""DOF index resolution for GenesisBinding."""

from __future__ import annotations

from typing import Any

import torch


class DOFResolver:
    """Helper class for resolving DOF indices."""

    def __init__(self, binding: Any):
        """Initialize the DOF resolver.

        Args:
            binding: Reference to the GenesisBinding instance.
        """
        self._binding = binding

    def resolve_dof_indices(self) -> None:
        """Resolve DOF indices for controlled joints in each robot."""
        for entity_name, robot_cfg in self._binding.cfg.robots.items():
            entity = self._binding._entities[entity_name]
            control_dofs = robot_cfg.control_dofs

            if control_dofs is None:
                # If no specific joints specified, get all actuated joints
                # This would require querying the entity for joint names
                # For now, we'll store None and resolve later when needed
                self._binding._dof_indices[entity_name] = None
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

                self._binding._dof_indices[entity_name] = torch.tensor(dof_indices, dtype=torch.long, device=self._binding.device)
