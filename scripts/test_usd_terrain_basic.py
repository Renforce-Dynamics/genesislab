#!/usr/bin/env python3
"""Basic test script for USD terrain support.

This script validates:
1. USD terrain loading from file
2. Environment origins generation (grid mode)
3. TerrainRuntime state initialization
4. Scene building without errors

Usage:
    python scripts/test_usd_terrain_basic.py
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


def test_usd_terrain_basic(usd_path: str, num_envs: int = 4, env_spacing: float = 5.0, viewer: bool = False):
    """Test basic USD terrain loading and initialization.

    Args:
        usd_path: Path to USD file to load.
        num_envs: Number of environments.
        env_spacing: Spacing between environments.
        viewer: Whether to show the viewer.
    """
    print("=" * 80)
    print("USD Terrain Basic Test")
    print("=" * 80)
    print(f"USD file: {usd_path}")
    print(f"Num envs: {num_envs}")
    print(f"Env spacing: {env_spacing}")
    print()

    # Validate USD file exists
    if not os.path.exists(usd_path):
        print(f"❌ ERROR: USD file not found: {usd_path}")
        return False
    print(f"✅ USD file found: {usd_path}")

    # Initialize Genesis
    print("\n📦 Initializing Genesis backend...")
    gs.init(backend=gs.cuda if torch.cuda.is_available() else gs.cpu)
    print(f"✅ Genesis initialized (backend: {'CUDA' if torch.cuda.is_available() else 'CPU'})")

    # Create scene configuration
    print("\n🏗️  Creating scene configuration...")
    scene_cfg = SceneCfg(
        num_envs=num_envs,
        env_spacing=(env_spacing, env_spacing),
        terrain=TerrainCfg(
            terrain_type="usd",
            usd_path=usd_path,
            env_spacing=env_spacing,
        ),
        viewer=viewer,
    )
    print("✅ Scene configuration created")

    # Create and build scene
    print("\n🎬 Building scene...")
    try:
        scene = LabScene(scene_cfg)
        print("✅ LabScene created")

        # Verify terrain was added
        if scene.terrain is None:
            print("❌ ERROR: Terrain was not created")
            return False
        print("✅ Terrain runtime created")

        # Validate terrain runtime state
        print("\n🔍 Validating terrain runtime state...")
        print(f"  - terrain_generator: {scene.terrain.terrain_generator}")
        print(f"  - terrain_origins: {scene.terrain.terrain_origins}")
        print(f"  - env_origins shape: {scene.terrain.env_origins.shape}")
        print(f"  - terrain_levels: {scene.terrain.terrain_levels}")
        print(f"  - terrain_types: {scene.terrain.terrain_types}")
        print(f"  - max_terrain_level: {scene.terrain.max_terrain_level}")

        # Validate env_origins
        if scene.terrain.env_origins is None:
            print("❌ ERROR: env_origins not created")
            return False

        expected_shape = (num_envs, 3)
        if scene.terrain.env_origins.shape != expected_shape:
            print(f"❌ ERROR: env_origins shape mismatch. Expected {expected_shape}, got {scene.terrain.env_origins.shape}")
            return False
        print(f"✅ env_origins shape correct: {scene.terrain.env_origins.shape}")

        # Print env origins
        print("\n📍 Environment origins:")
        for i, origin in enumerate(scene.terrain.env_origins):
            print(f"  Env {i}: [{origin[0]:7.2f}, {origin[1]:7.2f}, {origin[2]:7.2f}]")

        # Validate that this is grid mode (no curriculum)
        if scene.terrain.terrain_origins is not None:
            print("⚠️  WARNING: terrain_origins should be None for USD terrain (grid mode)")
        else:
            print("✅ Grid mode confirmed (terrain_origins is None)")

        if scene.terrain.terrain_levels is not None:
            print("⚠️  WARNING: terrain_levels should be None for USD terrain")
        else:
            print("✅ No curriculum (terrain_levels is None)")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

        if viewer:
            print("\n👁️  Viewer is open. Press Ctrl+C to exit...")
            try:
                while True:
                    scene.gs_scene.step()
            except KeyboardInterrupt:
                print("\n🛑 Viewer closed by user")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test USD terrain basic functionality")
    parser.add_argument(
        "--usd-path",
        type=str,
        default="third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd",
        help="Path to USD file",
    )
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--env-spacing", type=float, default=5.0, help="Environment spacing")
    parser.add_argument("--viewer", action="store_true", help="Show Genesis viewer")

    args = parser.parse_args()

    success = test_usd_terrain_basic(
        usd_path=args.usd_path,
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        viewer=args.viewer,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
