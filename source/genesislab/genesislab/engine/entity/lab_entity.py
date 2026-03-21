"""Entity data view for GenesisLab environments.

This module provides a data view abstraction similar to IsaacLab's ArticulationData,
allowing MDP code to access entity state through a clean, typed interface like
`env.entities["go2"].data.joint_pos` instead of calling `env.get_joint_state("go2")`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from genesislab.envs.manager_based_genesis_env import ManagerBasedGenesisEnv
    from genesislab.engine.gstype import KinematicEntity
    from genesislab.engine.assets.robot import Robot
    from genesislab.components.actuators import ActuatorBase
    from genesislab.engine.scene.actuator_manager import ActuatorManager

from .lab_entity_data import LabEntityData

class LabEntity:
    """Wrapper for a Genesis entity with data view access.

    This class wraps the underlying Genesis entity and provides a clean
    interface for accessing entity state through the `data` property,
    similar to IsaacLab's Articulation class.
    """

    _raw_entity: "KinematicEntity"
    _robot_asset: "Robot"
    _data: "LabEntityData" = None
    _actuators: Dict[str, "ActuatorBase"] = None
    _actuator_manager: "ActuatorManager | None" = None

    def __init__(self, env: "ManagerBasedGenesisEnv", entity_name: str, raw_entity: "KinematicEntity", robot_asset: "Robot" = None):
        """Initialize the entity wrapper.

        Args:
            env: The environment instance. Can be None if EntityData is not needed immediately.
            entity_name: Name of the entity.
            raw_entity: The underlying Genesis entity object.
            robot_asset: Optional Robot asset wrapper for name resolution (for articulated robots).
        """
        self._env = env
        self._entity_name = entity_name
        self._raw_entity = raw_entity
        self._robot_asset = robot_asset
        self._actuators = {}

    @property
    def actuator_manager(self) -> "ActuatorManager | None":
        """Per-robot actuator manager; set when the entity is added via :meth:`SceneBuilder.add_robot`."""
        return self._actuator_manager

    @property
    def entity_name(self) -> str:
        """Name of the entity."""
        return self._entity_name

    @property
    def name(self) -> str:
        """Name of the entity."""
        return self._entity_name

    @property
    def data(self) -> LabEntityData:
        """Data view for accessing entity state."""
        if self._data is None:
            # Pass self so that LabEntityData can access the raw entity directly
            self._data = LabEntityData(self._env, self)
        return self._data

    @property
    def raw_entity(self) -> KinematicEntity:
        """Access to the underlying Genesis entity (for advanced use cases)."""
        return self._raw_entity

    @property
    def robot_asset(self) -> Robot:
        """Access to the Robot asset wrapper (for name resolution, if available)."""
        return self._robot_asset

    @property
    def n_links(self):
        return self._raw_entity.n_links
    
    @property
    def raw_link_names(self):
        return self._robot_asset.body_normalizer.raw_names
    
    @property
    def link_names(self):
        return self._robot_asset.body_normalizer.normalized_names
    
    @property
    def n_joints(self):
        return self._raw_entity.n_joints
    
    @property
    def raw_joint_names(self):
        return self._robot_asset.joint_normalizer.raw_names
    
    @property
    def joint_names(self):
        return self._robot_asset.joint_normalizer.normalized_names
    
    @property
    def actuators(self) -> Dict[str, "ActuatorBase"]:
        """Dictionary of actuators for this entity, keyed by actuator name."""
        return self._actuators
    
    @property
    def dof_indices(self) -> torch.Tensor | None:
        """DOF indices for controlled joints.
        
        Returns:
            Tensor of DOF indices from the robot asset, or None if not available.
        """
        return self._robot_asset.dof_indices

    # ------------------------------------------------------------------
    # IsaacLab-style state write APIs used by imitation/tracking MDP
    # ------------------------------------------------------------------

    def write_joint_state_to_sim(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor | list[int] | None = None,
    ) -> None:
        # Infer DOF layout from entity
        dofs_pos_full = self._raw_entity.get_dofs_position()
        _, num_dofs_full = dofs_pos_full.shape
        base_offset = 6
        joint_slice = slice(base_offset, num_dofs_full)

        # Normalize env_ids to a 1D LongTensor
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device, dtype=torch.long)

        # Set joint DOFs only (exclude base)
        self._raw_entity.set_dofs_position(
            joint_pos,
            dofs_idx_local=joint_slice,
            envs_idx=env_ids,
            zero_velocity=False,
        )
        self._raw_entity.set_dofs_velocity(
            joint_vel,
            dofs_idx_local=joint_slice,
            envs_idx=env_ids,
        )

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor,
        root_vel: torch.Tensor,
        env_ids: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Write root state. Base is always 6 DOFs: pos(3) + Euler XYZ(3). Same convention as FK."""
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device, dtype=torch.long)

        pos = root_state[:, 0:3]
        rot = root_state[:, 3:6]
        lin_vel = root_vel[:, 0:3]
        ang_vel = root_vel[:, 3:6]

        base_q = torch.cat([pos, rot], dim=-1)
        self._raw_entity.set_dofs_position(
            base_q,
            dofs_idx_local=slice(0, 6),
            envs_idx=env_ids,
            zero_velocity=False,
        )
        base_vel = torch.cat([lin_vel, ang_vel], dim=-1)
        self._raw_entity.set_dofs_velocity(
            base_vel,
            dofs_idx_local=slice(0, 6),
            envs_idx=env_ids,
        )
