"""Entity data view for GenesisLab environments.

This module provides a data view abstraction similar to IsaacLab's ArticulationData,
allowing MDP code to access entity state through a clean, typed interface like
`env.entities["go2"].data.joint_pos` instead of calling `env.get_joint_state("go2")`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genesislab.envs.manager_based_genesis_env import ManagerBasedGenesisEnv

from .entity_data import EntityData

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
