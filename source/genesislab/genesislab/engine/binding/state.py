"""State query methods for GenesisBinding."""

from __future__ import annotations

from typing import Any

import torch


class StateQuerier:
    """Helper class for querying entity state."""

    def __init__(self, binding: Any):
        """Initialize the state querier.

        Args:
            binding: Reference to the GenesisBinding instance.
        """
        self._binding = binding

    def get_joint_state(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint positions and velocities for an entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            Tuple of (positions, velocities) tensors of shape (num_envs, num_dofs).
        """
        entity = self._binding._entities[entity_name]
        dof_indices = self._binding._dof_indices.get(entity_name)

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
        entity = self._binding._entities[entity_name]

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
        entity = self._binding._entities[entity_name]

        # Fast path: Genesis >= 0.3.x exposes a batched API.
        if hasattr(entity, "get_links_pos"):
            return entity.get_links_pos()

        # Alternative path: iterate over links if we can query them individually.
        # RigidEntity usually exposes ``n_links`` and ``get_link(idx)``.
        if not (hasattr(entity, "n_links") and hasattr(entity, "get_link")):
            raise AttributeError(
                f"Entity '{entity_name}' does not expose 'get_links_pos' or "
                f"('n_links' and 'get_link') methods; cannot fetch link positions."
            )
        
        num_links = int(entity.n_links)
        # Query first link to infer (num_envs, 3) shape and device
        if num_links == 0:
            return torch.zeros((self._binding.num_envs, 0, 3), device=self._binding.device)

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
