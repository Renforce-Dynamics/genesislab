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

        >>> # Track robot
        >>> camera_cfg = CameraCfg(
        ...     res=(1280, 720),
        ...     pos=(3.5, 0.0, 2.5),
        ...     lookat=(0.0, 0.0, 0.5),
        ...     entity_name="robot",
        ...     link_name="pelvis",
        ... )
    """

    # Resolution
    res: tuple[int, int] = (1280, 720)
    """Camera resolution (width, height)."""

    # Camera pose (world frame or entity-local)
    pos: tuple[float, float, float] = (3.5, 0.0, 2.5)
    """Camera position (x, y, z)."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    """Camera look-at target (x, y, z)."""

    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Camera up direction (x, y, z)."""

    fov: float = 40.0
    """Vertical field of view in degrees."""

    # Entity attachment (optional)
    entity_name: Optional[str] = None
    """Entity name to attach camera to. If None, camera is static in world frame."""

    link_name: Optional[str] = None
    """Link name to attach camera to (if entity_name is set). If None, attaches to entity root."""

    # Camera backend
    backend: Literal["rasterizer", "raytracer", "batch_renderer"] = "rasterizer"
    """Camera backend: 'rasterizer' (fast), 'raytracer' (high-quality), 'batch_renderer' (multi-env RL)."""

    # Display in viewer GUI
    show_in_gui: bool = False
    """Whether to show camera window in viewer GUI (only works when viewer=True)."""

    def __post_init__(self):
        """Validate configuration."""
        assert len(self.res) == 2, "res must be (width, height)"
        assert self.res[0] > 0 and self.res[1] > 0, "Resolution must be positive"
        assert self.fov > 0 and self.fov < 180, "FOV must be in (0, 180) degrees"


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
