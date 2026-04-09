#!/usr/bin/env python3
"""Demo script for Scene.usd terrain with visualization.

This script demonstrates USD terrain loading with the CWDL_LW_Assets Scene.usd file.
It creates a simple visualization with multiple environments placed on the terrain.

Usage:
    # Run with viewer (recommended)
    python scripts/demo_scene_usd_terrain.py --viewer

    # Quick test without viewer
    python scripts/demo_scene_usd_terrain.py

    # Custom USD file
    python scripts/demo_scene_usd_terrain.py --usd-path path/to/your/scene.usd --viewer
"""

import argparse
import os
import sys

import genesis as gs
import torch

# Add source to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source/genesislab"))

from genesislab.engine.scene import LabScene, SceneCfg, TerrainCfg


def demo_scene_usd_terrain(
    usd_path: str,
    num_envs: int = 9,
    env_spacing: float = 10.0,
    viewer: bool = True,
    run_steps: int = 500,
):
    """Run a demo with USD terrain visualization.

    Args:
        usd_path: Path to USD terrain file.
        num_envs: Number of environments (recommend 4, 9, or 16 for nice grid).
        env_spacing: Spacing between environments in meters.
        viewer: Whether to show the viewer.
        run_steps: Number of simulation steps to run.
    """
    print("=" * 80)
    print("🎬 SCENE.USD TERRAIN DEMO")
    print("=" * 80)
    print(f"📁 USD file: {usd_path}")
    print(f"🔢 Environments: {num_envs}")
    print(f"📏 Spacing: {env_spacing}m")
    print(f"👁️  Viewer: {'Enabled' if viewer else 'Disabled'}")
    print("=" * 80)

    # Check file exists
    if not os.path.exists(usd_path):
        print(f"\n❌ ERROR: USD file not found: {usd_path}")
        print(f"\n💡 Make sure the file exists at the specified path.")
        print(f"   Default path: third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd")
        return False

    print(f"\n✅ USD file found: {os.path.basename(usd_path)}")
    file_size = os.path.getsize(usd_path) / 1024  # KB
    print(f"   File size: {file_size:.1f} KB")

    # Initialize Genesis
    print("\n📦 Initializing Genesis...")
    backend = gs.cuda if torch.cuda.is_available() else gs.cpu
    gs.init(backend=backend)
    print(f"✅ Backend: {backend}")

    # Create scene configuration with USD terrain
    print("\n🏗️  Building scene with USD terrain...")
    scene_cfg = SceneCfg(
        num_envs=num_envs,
        env_spacing=(0.0, 0.0),  # Using terrain-based origins
        terrain=TerrainCfg(
            terrain_type="usd",
            usd_path=usd_path,
            env_spacing=env_spacing,
        ),
        viewer=viewer,
    )

    try:
        # Create scene
        scene = LabScene(scene_cfg)
        print("✅ Scene created successfully!")

        # Display terrain info
        print("\n📍 Terrain Information:")
        print(f"   Type: USD (static scene)")
        print(f"   Mode: Grid-based (no curriculum)")
        print(f"   Environment origins: {scene.terrain.env_origins.shape}")

        # Print environment layout
        print(f"\n🗺️  Environment Layout (3x3 grid at {env_spacing}m spacing):")
        grid_size = int(num_envs ** 0.5)
        for i in range(num_envs):
            origin = scene.terrain.env_origins[i]
            row = i // grid_size
            col = i % grid_size
            print(f"   Env {i:2d} [Row {row}, Col {col}]: "
                  f"X={origin[0]:7.2f}, Y={origin[1]:7.2f}, Z={origin[2]:7.2f}")

        if viewer:
            print("\n" + "=" * 80)
            print("👁️  VIEWER CONTROLS:")
            print("   - Mouse drag: Rotate camera")
            print("   - Mouse wheel: Zoom in/out")
            print("   - Arrow keys: Pan camera")
            print("   - Press Ctrl+C to exit")
            print("=" * 80)

            print(f"\n▶️  Running simulation for {run_steps} steps...")
            try:
                for step in range(run_steps):
                    scene.gs_scene.step()

                    if step % 100 == 0 and step > 0:
                        print(f"   Step {step}/{run_steps}...")

                print("\n✅ Simulation completed!")
                print("\n👁️  Viewer will stay open. Press Ctrl+C to exit...")

                # Keep viewer open
                while True:
                    scene.gs_scene.step()

            except KeyboardInterrupt:
                print("\n\n🛑 Demo stopped by user")

        else:
            print("\n⚠️  Viewer disabled. Running headless test...")
            for step in range(100):
                scene.gs_scene.step()
            print("✅ Headless test completed!")

        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Demo USD terrain with Scene.usd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with viewer (recommended)
  python scripts/demo_scene_usd_terrain.py --viewer

  # Run with 16 environments
  python scripts/demo_scene_usd_terrain.py --viewer --num-envs 16

  # Quick test without viewer
  python scripts/demo_scene_usd_terrain.py
        """,
    )
    parser.add_argument(
        "--usd-path",
        type=str,
        default="third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd",
        help="Path to USD terrain file",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=9,
        help="Number of environments (recommend 4, 9, or 16 for nice grid)",
    )
    parser.add_argument(
        "--env-spacing",
        type=float,
        default=10.0,
        help="Spacing between environments in meters",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show Genesis viewer (highly recommended!)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps to run before interactive mode",
    )

    args = parser.parse_args()

    # Run demo
    success = demo_scene_usd_terrain(
        usd_path=args.usd_path,
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        viewer=args.viewer,
        run_steps=args.steps,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
