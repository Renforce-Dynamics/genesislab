"""LabScene: Complete scene management for GenesisLab.

This module provides the LabScene class that manages all scene-related functionality,
including entities, sensors, scene construction, and coordination of query and control components.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Dict

import genesis as gs
import torch

if TYPE_CHECKING:
    from .lab_scene_cfg import SceneCfg
    from genesislab.components.sensors import SensorBase
    from genesislab.engine.entity import LabEntity
    from genesislab.engine.scene.terrain_runtime import TerrainRuntime
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv

from genesislab.engine.scene.scene_builder import SceneBuilder
from genesislab.engine.scene.scene_controller import SceneController


class LabScene:
    """Complete scene management for GenesisLab.
    
    This class manages:
    - Genesis Scene instance
    - Framework-managed entities and sensors
    - Scene construction and building
    - Coordination of query and control components
    """
    
    def __init__(self, cfg: "SceneCfg", device: str = "cuda"):
        """Initialize the LabScene.
        
        Args:
            cfg: Scene configuration.
            device: Device to use for tensors ('cuda' or 'cpu').
        """
        self.cfg = cfg
        # Derive device from cfg.backend when not explicitly overridden.
        # cfg.backend is 'cpu' or 'cuda'; if caller passes device explicitly it takes priority.
        if device == "cuda" and getattr(cfg, "backend", "cuda") == "cpu":
            device = "cpu"
        self.device = device
        self._gs_scene: gs.Scene = None
        self._entities: Dict[str, "LabEntity"] = {}
        self._sensors: Dict[str, "SensorBase"] = {}
        self._num_envs = cfg.num_envs
        self._terrain: TerrainRuntime | None = None
        
        # Initialize helper components
        self._scene_builder = SceneBuilder(self)
        self._controller = SceneController(self)
    
    @property
    def gs_scene(self) -> gs.Scene:
        """The underlying Genesis Scene instance."""
        if self._gs_scene is None:
            raise RuntimeError("Scene not built. Call build() first.")
        return self._gs_scene
    
    @property
    def entities(self) -> Dict[str, "LabEntity"]:
        """Dictionary of entities keyed by name."""
        return self._entities
    
    @property
    def sensors(self) -> Dict[str, "SensorBase"]:
        """Dictionary of sensors keyed by name."""
        return self._sensors
    
    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs
    
    @property
    def terrain(self) -> "TerrainRuntime | None":
        """The terrain runtime state, or ``None`` if no terrain is configured."""
        return self._terrain

    @property
    def env_origins(self) -> torch.Tensor | None:
        """Per-environment origins from terrain runtime.

        Returns ``None`` if no terrain runtime is available.
        """
        if self._terrain is None:
            return None
        return self._terrain.env_origins

    @property
    def controller(self) -> "SceneController":
        """Scene controller for control and state setting."""
        return self._controller
    
    def build(self, env: Any = None) -> None:
        """Build the Genesis scene and entities.
        
        This method:
        1. Creates a Genesis Scene with appropriate options
        2. Adds robots and terrain according to cfg
        3. Builds the scene with num_envs
        4. Constructs LabEntity objects for each robot (each with its own :class:`~genesislab.engine.scene.actuator_manager.ActuatorManager`)
        5. After ``scene.build()`` runs each entity's actuator manager to create actuators and sync engine PD gains
        
        Args:
            env: Optional environment instance (ManagerBasedGenesisEnv). 
                Required for constructing LabEntity.
        """
        # Create Genesis scene
        self._gs_scene = self._scene_builder.create_scene()
        
        # Add terrain if specified — stores the TerrainRuntime
        if self.cfg.terrain is not None:
            self._terrain = self._scene_builder.add_terrain(self._gs_scene)

        # Add robots
        for entity_name, robot_cfg in self.cfg.robots.items():
            lab_entity = self._scene_builder.add_robot(self._gs_scene, entity_name, robot_cfg, env=env)
            self._entities[entity_name] = lab_entity
        
        # Add sensors if specified
        for sensor_name, sensor_cfg in self.cfg.sensors.items():
            self._scene_builder.add_sensor(self, sensor_name, sensor_cfg)
        
        # Optional: attach a simple camera and start video recording
        video_path = getattr(self.cfg, "record_video_path", None)
        if video_path is not None:
            from genesislab.engine.visualize import attach_video_recorder
            attach_video_recorder(self._gs_scene, str(video_path))
        
        # Build the scene
        self._scene_builder.build_scene(self._gs_scene)

        # Per-robot setup
        for entity_name, lab_entity in self._entities.items():
            if lab_entity.actuator_manager is not None:
                lab_entity.actuator_manager.setup(self.cfg.robots[entity_name])
    
    def add_entity(self, name: str, entity: "LabEntity") -> None:
        """Add an entity to the scene.
        
        Args:
            name: Entity name.
            entity: LabEntity instance.
        """
        self._entities[name] = entity
    
    def add_sensor(self, name: str, sensor: "SensorBase") -> None:
        """Add a sensor to the scene.
        
        Args:
            name: Sensor name.
            sensor: Sensor instance.
        """
        self._sensors[name] = sensor
    
    def get_sensor(self, name: str) -> "SensorBase":
        """Get a sensor by name.
        
        Args:
            name: Sensor name.
            
        Returns:
            Sensor instance.
            
        Raises:
            KeyError: If sensor not found.
        """
        if name not in self._sensors:
            raise KeyError(
                f"Sensor '{name}' not found. "
                f"Available sensors: {list(self._sensors.keys())}"
            )
        return self._sensors[name]
    
    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to the underlying Genesis Scene.
        
        This allows LabScene to be used as a drop-in replacement for
        the Genesis Scene object while managing framework-internal objects
        separately.
        
        Args:
            name: Attribute name.
            
        Returns:
            Attribute value from the Genesis Scene.
        """
        return getattr(self._gs_scene, name)
