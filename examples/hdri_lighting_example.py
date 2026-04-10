"""Example: Using HDRI Environment Lighting in GenesisLab

This example demonstrates how to configure HDRI environment lighting (IBL)
for realistic lighting in your scene.

Before running:
1. Place your sky.hdr file in: data/assets/hdri/sky.hdr
2. Or download a free HDRI from https://polyhaven.com/hdris
"""

from genesislab.engine.scene import SceneCfg
from genesislab.engine.sim import VisOptionsCfg, ViewerOptionsCfg, SimOptionsCfg
from genesislab.components.terrains import TerrainCfg

# Example 1: Basic HDRI environment lighting
scene_cfg_with_hdri = SceneCfg(
    num_envs=1,
    viewer=True,
    viewer_options=ViewerOptionsCfg(
        camera_pos=(3.0, 3.0, 2.0),
        camera_lookat=(0.0, 0.0, 0.5),
    ),
    vis_options=VisOptionsCfg(
        # Enable HDRI environment lighting
        env_surface="sky.hdr",  # Looks for data/assets/hdri/sky.hdr
        env_radius=1000.0,
        env_pos=(0.0, 0.0, 0.0),
    ),
    sim_options=SimOptionsCfg(
        dt=0.01,
        gravity=(0.0, 0.0, -9.81),
    ),
    terrain=TerrainCfg(
        terrain_type="plane",
    ),
)

# Example 2: HDRI with custom lights (fallback for rasterizer)
scene_cfg_with_lights = SceneCfg(
    num_envs=1,
    viewer=True,
    vis_options=VisOptionsCfg(
        # HDRI for high-quality rendering
        env_surface="sky.hdr",
        env_radius=1000.0,

        # Additional directional lights for real-time viewer
        lights=[
            {
                "type": "directional",
                "dir": (-1.0, -1.0, -2.0),
                "color": (1.0, 1.0, 0.95),  # Slightly warm white
                "intensity": 5.0,
            },
            {
                "type": "directional",
                "dir": (1.0, 0.5, -1.0),
                "color": (0.7, 0.8, 1.0),  # Cool blue fill light
                "intensity": 2.0,
            },
        ],
        # Ambient light for softer shadows
        ambient_light=(0.3, 0.3, 0.3),

        # Background color
        background_color=(0.5, 0.7, 1.0),  # Sky blue
    ),
    terrain=TerrainCfg(
        terrain_type="plane",
    ),
)

# Example 3: Using absolute path for HDRI
scene_cfg_absolute_path = SceneCfg(
    num_envs=1,
    viewer=True,
    vis_options=VisOptionsCfg(
        # Use absolute path for HDRI file
        env_surface="/path/to/your/custom.hdr",  # Replace with actual path
        env_radius=1000.0,
    ),
)


if __name__ == "__main__":
    print("HDRI Lighting Configuration Examples")
    print("=" * 50)
    print("\nExample 1: Basic HDRI")
    print(f"  env_surface: {scene_cfg_with_hdri.vis_options.env_surface}")
    print(f"  env_radius: {scene_cfg_with_hdri.vis_options.env_radius}")

    print("\nExample 2: HDRI with custom lights")
    print(f"  env_surface: {scene_cfg_with_lights.vis_options.env_surface}")
    print(f"  lights: {len(scene_cfg_with_lights.vis_options.lights)} lights")
    print(f"  ambient_light: {scene_cfg_with_lights.vis_options.ambient_light}")

    print("\nTo use these configurations in your environment:")
    print("  env = YourEnv(scene_cfg=scene_cfg_with_hdri)")
    print("\nMake sure sky.hdr is placed in: data/assets/hdri/sky.hdr")
