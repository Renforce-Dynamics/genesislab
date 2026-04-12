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
        self._environment_objects: Dict[str, object] = {}
        self._num_envs = cfg.num_envs
        self._terrain: TerrainRuntime | None = None
        
        # Initialize helper components
        self._scene_builder = SceneBuilder(self)
        self._controller = SceneController(self)
    
        self._registries = [
            ("entity", self._entities),
            ("environment_object", self._environment_objects),
            ("sensor", self._sensors)
        ]
    
    def __getitem__(self, key):
        for registry_name, registry in self._registries:
            if key in registry:
                return registry[key]
        raise KeyError(
            f"{key!r} not found in LabScene. "
            f"Available keys:\n"
            f"  entities: {list(self._entities.keys())}\n"
            f"  env_objects: {list(self._environment_objects.keys())}\n"
            f"  sensors: {list(self._sensors.keys())}"
        )
    
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
    def environment_objects(self) -> Dict[str, object]:
        """Dictionary of environment objects keyed by name.

        Environment objects are interactive scene elements (furniture, props, etc.)
        that robots can interact with. Access object state via Genesis entity API.

        Example:
            >>> objects = env.scene.environment_objects
            >>> if "chair_01" in objects:
            ...     chair = objects["chair_01"]
            ...     chair_pos = chair.get_pos()
        """
        return self._environment_objects

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
        4. Constructs LabEntity objects for each robot
        5. ``scene.build()``, then :meth:`~genesislab.engine.assets.robot.robot.Robot.initialize_actuators`
           on each robot (needs a built scene for DOF queries)
        
        Args:
            env: Optional environment instance (ManagerBasedGenesisEnv). 
                Required for constructing LabEntity.
        """
        # Create Genesis scene
        self._gs_scene = self._scene_builder.create_scene()

        # Add USD scene if specified (background environment with furniture, buildings, etc.)
        if self.cfg.usd_scene_path is not None:
            self._scene_builder.add_usd_scene(self._gs_scene)

        # Add terrain if specified — stores the TerrainRuntime
        if self.cfg.terrain is not None:
            self._terrain = self._scene_builder.add_terrain(self._gs_scene)

        # Add robots (defines control DOF space)
        for entity_name, robot_cfg in self.cfg.robots.items():
            lab_entity = self._scene_builder.add_robot(self._gs_scene, entity_name, robot_cfg, env=env)
            self._entities[entity_name] = lab_entity

        # Add environment objects (after robots, to avoid joint indexing conflicts)
        if self.cfg.objects:
            self._environment_objects = self._scene_builder.add_environment_objects(
                self._gs_scene,
                self.cfg.objects
            )

        # Add sensors if specified
        for sensor_name, sensor_cfg in self.cfg.sensors.items():
            self._scene_builder.add_sensor(self, sensor_name, sensor_cfg)
            
        # Optional: attach camera and start video recording (after scene.build())
        # New camera/recording API (preferred)
        if self.cfg.camera is not None:
            self._setup_camera_and_recording()
        # Legacy API (deprecated, but still supported)
        elif getattr(self.cfg, "record_video_path", None) is not None:
            from genesislab.engine.visualize import attach_video_recorder
            attach_video_recorder(self._gs_scene, str(self.cfg.record_video_path))

        # Build the scene (required before DOF / actuator setup on entities)
        self._scene_builder.build_scene(self._gs_scene)

        for lab_entity in self._entities.values():
            lab_entity.robot_asset.initialize_actuators()

    def _setup_camera_and_recording(self) -> None:
        """Setup camera and optional video recording based on configuration.

        This method is called after scene.build() to add a camera and
        optionally start video recording in headless mode.
        """
        from pathlib import Path

        cam_cfg = self.cfg.camera
        rec_cfg = self.cfg.recording

        # Determine entity attachment
        entity_idx = -1  # Default: static camera
        link_idx_local = 0

        if cam_cfg.entity_name is not None:
            # Find entity
            if cam_cfg.entity_name not in self._entities:
                raise ValueError(
                    f"Camera entity '{cam_cfg.entity_name}' not found. "
                    f"Available entities: {list(self._entities.keys())}"
                )
            entity = self._entities[cam_cfg.entity_name]
            entity_idx = entity.gs_entity.idx

            # Find link if specified
            if cam_cfg.link_name is not None:
                link_names = entity.gs_entity.links_map
                if cam_cfg.link_name not in link_names:
                    raise ValueError(
                        f"Camera link '{cam_cfg.link_name}' not found in entity '{cam_cfg.entity_name}'. "
                        f"Available links: {list(link_names.keys())}"
                    )
                link_idx_local = link_names[cam_cfg.link_name]

        # Create camera based on backend
        if cam_cfg.backend == "rasterizer":
            camera = self._gs_scene.add_camera(
                res=cam_cfg.res,
                pos=cam_cfg.pos,
                lookat=cam_cfg.lookat,
                up=cam_cfg.up,
                fov=cam_cfg.fov,
                GUI=cam_cfg.show_in_gui,
            )
        elif cam_cfg.backend == "raytracer":
            # Raytracer requires RayTracer renderer to be set in scene creation
            camera = self._gs_scene.add_sensor(
                gs.sensors.RaytracerCameraOptions(
                    res=cam_cfg.res,
                    pos=cam_cfg.pos,
                    lookat=cam_cfg.lookat,
                    up=cam_cfg.up,
                    fov=cam_cfg.fov,
                    entity_idx=entity_idx,
                    link_idx_local=link_idx_local,
                )
            )
        elif cam_cfg.backend == "batch_renderer":
            # BatchRenderer requires BatchRenderer to be set in scene creation
            camera = self._gs_scene.add_sensor(
                gs.sensors.BatchRendererCameraOptions(
                    res=cam_cfg.res,
                    pos=cam_cfg.pos,
                    lookat=cam_cfg.lookat,
                    up=cam_cfg.up,
                    fov=cam_cfg.fov,
                    entity_idx=entity_idx,
                    link_idx_local=link_idx_local,
                )
            )
        else:
            raise ValueError(f"Unknown camera backend: {cam_cfg.backend}")

        # Store camera reference
        self._camera = camera

        # Start recording if configured
        if rec_cfg is not None and rec_cfg.enabled:
            # Create output directory
            save_path = Path(rec_cfg.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Build codec options
            codec_options = {
                "preset": rec_cfg.codec_preset,
            }
            if rec_cfg.codec_tune is not None:
                codec_options["tune"] = rec_cfg.codec_tune

            # Start recording
            self._gs_scene.start_recording(
                data_func=lambda: camera.render(
                    rgb=rec_cfg.render_rgb,
                    depth=rec_cfg.render_depth,
                    segmentation=rec_cfg.render_segmentation,
                    normal=rec_cfg.render_normal,
                )[0],  # [0] extracts RGB from tuple
                rec_options=gs.recorders.VideoFile(
                    filename=str(save_path),
                    fps=rec_cfg.fps,
                    codec=rec_cfg.codec,
                    codec_options=codec_options,
                ),
            )
            print(f"[LabScene] Started video recording: {save_path}")
            print(f"           Camera: {cam_cfg.res[0]}x{cam_cfg.res[1]}, {rec_cfg.fps} FPS")

    @property
    def camera(self):
        """Get the scene camera if configured, otherwise None."""
        return getattr(self, "_camera", None)

    def render_camera(self, rgb=True, depth=False, segmentation=False, normal=False):
        """Manually render the scene camera.

        This is useful for getting camera output without recording.
        If recording is enabled, camera rendering happens automatically.

        Args:
            rgb: Render RGB image.
            depth: Render depth image.
            segmentation: Render segmentation mask.
            normal: Render normal map.

        Returns:
            Tuple of rendered outputs (rgb, depth, segmentation, normal).
            Non-requested outputs are None.

        Raises:
            RuntimeError: If camera is not configured.
        """
        if self.camera is None:
            raise RuntimeError(
                "Camera not configured. Set camera=CameraCfg(...) in SceneCfg to enable camera."
            )
        return self.camera.render(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            normal=normal,
        )

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
