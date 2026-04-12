"""Camera and recording configuration for GenesisLab scenes."""

from __future__ import annotations

from typing import Literal, Optional

from genesislab.utils.configclass import configclass


@configclass
class CameraCfg:
    """Configuration for scene camera in headless rendering.

    This camera can be used for headless rendering and video recording.
    If not configured (camera=None in SceneCfg), no camera will be added.

    Example:
        >>> # Static camera
        >>> camera_cfg = CameraCfg(
        ...     res=(1280, 720),
        ...     pos=(3.5, 0.0, 2.5),
        ...     lookat=(0.0, 0.0, 0.5),
        ...     fov=40,
        ... )

        >>> # Track robot (chase camera)
        >>> camera_cfg = CameraCfg(
        ...     res=(1280, 720),
        ...     track_mode="chase",
        ...     entity_name="robot",
        ...     fov=45,
        ... )

        >>> # Custom entity-attached camera
        >>> camera_cfg = CameraCfg(
        ...     res=(1280, 720),
        ...     entity_name="robot",
        ...     link_name="pelvis",
        ...     pos=(2.0, 0.0, 1.5),     # Offset from link in link's local frame
        ...     lookat=(5.0, 0.0, 0.5),  # Look direction in link's local frame
        ... )
    """

    # Resolution
    res: tuple[int, int] = (1280, 720)
    """Camera resolution (width, height)."""

    # Tracking mode (convenient presets)
    track_mode: Optional[Literal["static", "chase", "follow", "side", "top", "first_person"]] = None
    """Camera tracking mode preset:
    - None: Use manual pos/lookat configuration
    - "static": Static camera (world frame, ignores entity_name)
    - "chase": Chase camera behind and above robot (cinemati c view)
    - "follow": Follow camera directly behind robot (over-the-shoulder)
    - "side": Side view following robot
    - "top": Top-down view following robot
    - "first_person": First-person view from robot head/body

    When track_mode is set, pos/lookat are automatically configured.
    You can still override them if needed.
    """

    # Camera pose (world frame or entity-local if entity_name is set)
    pos: tuple[float, float, float] = (3.5, 0.0, 2.5)
    """Camera position (x, y, z).
    - If entity_name=None: world frame position
    - If entity_name is set: offset from entity link in link's local frame
    """

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    """Camera look-at target (x, y, z).
    - If entity_name=None: world frame position
    - If entity_name is set: target offset in link's local frame (usually forward direction)
    """

    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Camera up direction (x, y, z)."""

    fov: float = 40.0
    """Vertical field of view in degrees."""

    # Entity attachment (for tracking robot)
    entity_name: Optional[str] = None
    """Entity name to attach camera to. If None, camera is static in world frame.
    Set this to "robot" (or your robot entity name) to make camera follow the robot.
    """

    link_name: Optional[str] = None
    """Link name to attach camera to (if entity_name is set).
    - If None: attaches to entity root
    - Common choices: "pelvis", "base_link", "torso", etc.
    """

    # Camera backend
    backend: Literal["rasterizer", "raytracer", "batch_renderer"] = "rasterizer"
    """Camera backend: 'rasterizer' (fast), 'raytracer' (high-quality), 'batch_renderer' (multi-env RL)."""

    # Display in viewer GUI
    show_in_gui: bool = False
    """Whether to show camera window in viewer GUI (only works when viewer=True)."""

    def __post_init__(self):
        """Validate and apply tracking mode presets."""
        assert len(self.res) == 2, "res must be (width, height)"
        assert self.res[0] > 0 and self.res[1] > 0, "Resolution must be positive"
        assert self.fov > 0 and self.fov < 180, "FOV must be in (0, 180) degrees"

        # Apply tracking mode presets if specified
        if self.track_mode is not None:
            self._apply_track_mode()

    def _apply_track_mode(self):
        """Apply camera position/lookat based on tracking mode preset."""
        # Track mode presets define camera offset and look direction in entity's local frame
        # These are designed for humanoid robots (like G1)

        track_presets = {
            "static": {
                # Static camera in world frame (ignores entity attachment)
                "pos": (5.0, 0.0, 3.0),
                "lookat": (0.0, 0.0, 0.5),
                "entity_name": None,  # Force static
                "fov": 45.0,
            },
            "chase": {
                # Cinematic chase camera: behind and above robot
                "pos": (-3.5, 0.0, 2.5),      # Behind (-X), elevated (+Z)
                "lookat": (1.0, 0.0, 0.5),    # Look forward (+X)
                "fov": 50.0,
            },
            "follow": {
                # Over-the-shoulder follow camera: closer, directly behind
                "pos": (-2.0, 0.5, 1.8),      # Behind, slightly to side
                "lookat": (1.0, 0.0, 0.8),    # Look forward, slightly up
                "fov": 45.0,
            },
            "side": {
                # Side view: perpendicular to robot
                "pos": (0.0, -3.0, 1.5),      # To the side (-Y)
                "lookat": (0.0, 0.5, 0.8),    # Look slightly inward (+Y)
                "fov": 45.0,
            },
            "top": {
                # Top-down view: directly above robot
                "pos": (0.0, 0.0, 5.0),       # Above (+Z)
                "lookat": (0.5, 0.0, -1.0),   # Look down (-Z), slightly forward
                "fov": 60.0,
            },
            "first_person": {
                # First-person view from robot's head/body
                "pos": (0.0, 0.0, 0.2),       # Slightly above link origin
                "lookat": (1.0, 0.0, 0.0),    # Look straight forward (+X)
                "link_name": "pelvis",        # Attach to pelvis or head
                "fov": 75.0,
            },
        }

        if self.track_mode not in track_presets:
            raise ValueError(
                f"Unknown track_mode: {self.track_mode}. "
                f"Available modes: {list(track_presets.keys())}"
            )

        preset = track_presets[self.track_mode]

        # Apply preset values (only if not manually overridden)
        # We check if values are still at defaults before overriding
        if self.pos == (3.5, 0.0, 2.5):  # Default pos
            self.pos = preset["pos"]
        if self.lookat == (0.0, 0.0, 0.5):  # Default lookat
            self.lookat = preset["lookat"]
        if "fov" in preset and self.fov == 40.0:  # Default fov
            self.fov = preset["fov"]
        if "link_name" in preset and self.link_name is None:
            self.link_name = preset["link_name"]
        if "entity_name" in preset:
            self.entity_name = preset["entity_name"]


@configclass
class RecordingCfg:
    """Configuration for video recording from camera.

    If configured, recording will start automatically during scene build
    and can be stopped manually with scene.gs_scene.stop_recording().

    Example:
        >>> recording_cfg = RecordingCfg(
        ...     enabled=True,
        ...     save_path="output/video.mp4",
        ...     fps=60,
        ... )
    """

    enabled: bool = False
    """Whether to enable video recording."""

    save_path: str = "output/recording.mp4"
    """Output video file path. Parent directories will be created automatically."""

    fps: int = 60
    """Video frame rate (frames per second)."""

    codec: str = "libx264"
    """Video codec. Common options: 'libx264' (H.264), 'libx265' (H.265/HEVC)."""

    codec_preset: Literal["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"] = "veryfast"
    """Codec preset controlling encoding speed vs compression.
    - 'ultrafast', 'veryfast': Fast encoding, larger file
    - 'medium': Balanced
    - 'slow', 'veryslow': Slow encoding, smaller file
    """

    codec_tune: Optional[Literal["film", "animation", "grain", "stillimage", "fastdecode", "zerolatency"]] = "zerolatency"
    """Codec tuning option:
    - 'zerolatency': Low latency (good for real-time encoding)
    - 'film': Live-action content
    - 'animation': Animated content
    - None: No tuning
    """

    render_rgb: bool = True
    """Whether to render RGB channels."""

    render_depth: bool = False
    """Whether to render depth channel (not used for standard video)."""

    render_segmentation: bool = False
    """Whether to render segmentation (not used for standard video)."""

    render_normal: bool = False
    """Whether to render normal map (not used for standard video)."""

    def __post_init__(self):
        """Validate configuration."""
        assert self.fps > 0, "FPS must be positive"
        if self.enabled:
            assert self.save_path, "save_path must be specified when recording is enabled"
            # Ensure save_path ends with .mp4
            if not self.save_path.endswith(('.mp4', '.MP4')):
                print(f"[RecordingCfg] Warning: save_path should end with .mp4, got {self.save_path}")
