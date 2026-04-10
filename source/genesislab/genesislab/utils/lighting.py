"""Lighting utilities for GenesisLab scenes.

This module provides helper functions for configuring scene lighting,
including HDRI environment lighting (IBL).
"""

import os
from typing import Optional

from genesislab.engine.sim import VisOptionsCfg


def get_default_hdri_path() -> str:
    """Get the default path to the sky.hdr HDRI file.

    Returns:
        Absolute path to sky.hdr in data/assets/hdri/.
    """
    # Get the directory where this file (lighting.py) is located
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to genesislab package root: utils -> genesislab -> genesislab (source)
    genesislab_dir = os.path.dirname(utils_dir)
    source_dir = os.path.dirname(genesislab_dir)
    # Go up to project root (where data/ is located)
    project_root = os.path.dirname(os.path.dirname(source_dir))
    return os.path.join(project_root, "data", "assets", "hdri", "sky.hdr")


def hdri_exists(hdri_name: str = "sky.hdr") -> bool:
    """Check if an HDRI file exists in the default directory.

    Args:
        hdri_name: Name of the HDRI file (default: "sky.hdr")

    Returns:
        True if the file exists, False otherwise.
    """
    # Get the directory where this file (lighting.py) is located
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to genesislab package root: utils -> genesislab -> genesislab (source)
    genesislab_dir = os.path.dirname(utils_dir)
    source_dir = os.path.dirname(genesislab_dir)
    # Go up to project root (where data/ is located)
    project_root = os.path.dirname(os.path.dirname(source_dir))
    hdri_path = os.path.join(project_root, "data", "assets", "hdri", hdri_name)
    return os.path.exists(hdri_path)


def create_vis_options_with_hdri(
    hdri_name: str = "sky.hdr",
    env_radius: float = 1000.0,
    env_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    add_lights: bool = True,
    ambient_light: Optional[tuple[float, float, float]] = None,
    background_color: Optional[tuple[float, float, float]] = None,
    enable_hdri: bool = True,
) -> VisOptionsCfg:
    """Create a VisOptionsCfg with HDRI environment lighting.

    This is a convenience function to quickly set up HDRI-based lighting
    for your scene. It will check if the HDRI file exists and configure
    the visualization options accordingly.

    Note:
        HDRI environment lighting requires LuisaRenderPy to be installed.
        If not available, the system will automatically fall back to
        directional lights. Install with: pip install luisa-python

    Args:
        hdri_name: Name of the HDRI file in data/assets/hdri/ (default: "sky.hdr")
        env_radius: Radius of the environment sphere (default: 1000.0)
        env_pos: Position of the environment sphere (default: (0, 0, 0))
        add_lights: Whether to add directional lights for rasterizer fallback (default: True)
        ambient_light: Ambient light color (r, g, b). If None and add_lights=True,
                      uses (0.3, 0.3, 0.3)
        background_color: Background color (r, g, b). If None, uses Genesis default
        enable_hdri: Whether to enable HDRI (default: True). Set to False to only
                    use directional lights without attempting HDRI.

    Returns:
        VisOptionsCfg with HDRI environment lighting configured

    Example:
        >>> vis_options = create_vis_options_with_hdri()
        >>> scene_cfg = SceneCfg(vis_options=vis_options)
    """
    # Check if HDRI should be enabled
    if not enable_hdri:
        hdri_name = None
    elif not hdri_exists(hdri_name):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"HDRI file '{hdri_name}' not found in data/assets/hdri/. "
            f"HDRI environment lighting will be disabled. "
            f"Download HDRI files from https://polyhaven.com/hdris"
        )
        hdri_name = None

    # Create default directional lights for rasterizer
    lights = None
    if add_lights:
        lights = [
            {
                "type": "directional",
                "dir": (-1.0, -1.0, -2.0),
                "color": (1.0, 1.0, 0.95),  # Slightly warm white (key light)
                "intensity": 5.0,
            },
            {
                "type": "directional",
                "dir": (1.0, 0.5, -1.0),
                "color": (0.7, 0.8, 1.0),  # Cool blue fill light
                "intensity": 2.0,
            },
        ]
        if ambient_light is None:
            ambient_light = (0.3, 0.3, 0.3)

    return VisOptionsCfg(
        env_surface=hdri_name,
        env_radius=env_radius,
        env_pos=env_pos,
        lights=lights,
        ambient_light=ambient_light,
        background_color=background_color,
    )


def create_studio_lighting(
    key_intensity: float = 5.0,
    fill_intensity: float = 2.0,
    ambient: tuple[float, float, float] = (0.3, 0.3, 0.3),
    background: tuple[float, float, float] = (0.5, 0.7, 1.0),
) -> VisOptionsCfg:
    """Create a VisOptionsCfg with studio-style three-point lighting.

    This creates a classic three-point lighting setup with a key light,
    fill light, and ambient lighting. Good for when HDRI is not available.

    Args:
        key_intensity: Intensity of the main key light (default: 5.0)
        fill_intensity: Intensity of the fill light (default: 2.0)
        ambient: Ambient light color (r, g, b) (default: (0.3, 0.3, 0.3))
        background: Background color (r, g, b) (default: sky blue)

    Returns:
        VisOptionsCfg with studio lighting configured

    Example:
        >>> vis_options = create_studio_lighting()
        >>> scene_cfg = SceneCfg(vis_options=vis_options)
    """
    lights = [
        # Key light (main directional light from above-front-left)
        {
            "type": "directional",
            "dir": (-1.0, -1.0, -2.0),
            "color": (1.0, 1.0, 0.95),  # Slightly warm white
            "intensity": key_intensity,
        },
        # Fill light (softer light from right to reduce shadows)
        {
            "type": "directional",
            "dir": (1.0, 0.5, -1.0),
            "color": (0.7, 0.8, 1.0),  # Cool blue
            "intensity": fill_intensity,
        },
        # Back/rim light (from behind to separate subject from background)
        {
            "type": "directional",
            "dir": (0.5, 1.0, -0.5),
            "color": (1.0, 0.95, 0.9),  # Warm white
            "intensity": fill_intensity * 0.8,
        },
    ]

    return VisOptionsCfg(
        lights=lights,
        ambient_light=ambient,
        background_color=background,
    )
