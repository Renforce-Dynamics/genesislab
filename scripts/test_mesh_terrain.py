#!/usr/bin/env python3
"""Simple test for mesh terrain loading.

Tests that GLB/OBJ/STL/GLTF files can be loaded as terrain.

Usage:
    python scripts/test_mesh_terrain.py
    python scripts/test_mesh_terrain.py --mesh-path data/assets/Barracks.glb
"""

import argparse
import genesis as gs
from genesislab.components.terrains import TerrainCfg
from genesislab.engine.scene import LabScene, SceneCfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", type=str, default="data/assets/city/source/Untitled.glb")
    parser.add_argument("--viewer", action="store_true", help="Enable viewer")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--sdf-cell-size", type=float, default=0.2, help="SDF cell size (larger = less memory)")
    args = parser.parse_args()

    print(f"\n🧪 Testing mesh terrain loading...")
    print(f"   Mesh file: {args.mesh_path}")
    print(f"   Num envs: {args.num_envs}")
    print(f"   Viewer: {args.viewer}")

    # Initialize Genesis
    gs.init(backend=gs.cuda)

    # Create scene config with mesh terrain
    scene_cfg = SceneCfg(
        num_envs=args.num_envs,
        backend="cuda",
        viewer=args.viewer,
        terrain=TerrainCfg(
            terrain_type="mesh",
            mesh_path=args.mesh_path,
            env_spacing=3.0,
            mesh_decompose_error_threshold=float("inf"),  # Skip decomposition for speed
            mesh_sdf_cell_size=args.sdf_cell_size,  # Larger for big meshes like cities
        ),
    )

    # Build scene
    print("\n📦 Building scene with mesh terrain...")
    scene = LabScene(cfg=scene_cfg)
    scene.build()

    print("\n✅ Mesh terrain loaded successfully!")
    print(f"   Scene: {scene._gs_scene}")
    print(f"   Terrain runtime: {scene.terrain}")
    print(f"   Env origins shape: {scene.terrain.env_origins.shape}")

    # Optional: step simulation to verify physics
    if args.viewer:
        print("\n▶️  Running simulation for 100 steps...")
        for i in range(1000):
            scene._gs_scene.step()
            if i % 20 == 0:
                print(f"   Step {i}/100")

    print("\n✅ Test passed!")


if __name__ == "__main__":
    main()
