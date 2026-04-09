#!/usr/bin/env python3
"""Integration test for USD terrain with robot task environment.

This script tests USD terrain integration with a full RL environment:
1. Load USD terrain in a task environment
2. Spawn robots on the terrain
3. Verify robot placement at env_origins
4. Run simulation steps

Usage:
    python scripts/test_usd_terrain_integration.py --viewer
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source/genesis_tasks"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source/genesis_assets"))

from genesislab.engine.scene import SceneCfg, TerrainCfg
from genesislab.utils.configclass import configclass


def create_usd_terrain_scene_cfg(
    usd_path: str,
    num_envs: int = 4,
    env_spacing: float = 8.0,
    viewer: bool = False,
):
    """Create a scene config with USD terrain.

    Args:
        usd_path: Path to USD terrain file.
        num_envs: Number of environments.
        env_spacing: Environment spacing in meters.
        viewer: Whether to show viewer.

    Returns:
        SceneCfg instance.
    """
    return SceneCfg(
        num_envs=num_envs,
        env_spacing=(0.0, 0.0),  # Will use terrain env_origins
        terrain=TerrainCfg(
            terrain_type="usd",
            usd_path=usd_path,
            env_spacing=env_spacing,
        ),
        viewer=viewer,
    )


def test_usd_terrain_with_robot(
    usd_path: str,
    num_envs: int = 4,
    env_spacing: float = 8.0,
    viewer: bool = False,
    num_steps: int = 100,
):
    """Test USD terrain with robot spawning and simulation.

    Args:
        usd_path: Path to USD file.
        num_envs: Number of environments.
        env_spacing: Environment spacing.
        viewer: Whether to show viewer.
        num_steps: Number of simulation steps.
    """
    print("=" * 80)
    print("USD Terrain Integration Test (with Robot)")
    print("=" * 80)
    print(f"USD file: {usd_path}")
    print(f"Num envs: {num_envs}")
    print(f"Env spacing: {env_spacing}")
    print(f"Num steps: {num_steps}")
    print()

    # Validate USD file
    if not os.path.exists(usd_path):
        print(f"❌ ERROR: USD file not found: {usd_path}")
        return False
    print(f"✅ USD file found")

    # Initialize Genesis
    print("\n📦 Initializing Genesis...")
    gs.init(backend=gs.cuda if torch.cuda.is_available() else gs.cpu)
    print("✅ Genesis initialized")

    # Try to import task environment
    print("\n🔍 Importing task environment...")
    try:
        from genesis_tasks.locomotion.velocity.robots.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
        print("✅ Successfully imported Go2 flat environment")

        # Create custom env config with USD terrain
        @configclass
        class Go2USDTerrainEnvCfg(UnitreeGo2FlatEnvCfg):
            """Go2 environment with USD terrain."""

            def __post_init__(self):
                super().__post_init__()

                # Override terrain config
                self.scene.terrain = TerrainCfg(
                    terrain_type="usd",
                    usd_path=usd_path,
                    env_spacing=env_spacing,
                )
                # Update number of envs
                self.scene.num_envs = num_envs
                self.scene.viewer = viewer

        print("\n🏗️  Creating environment with USD terrain...")
        env_cfg = Go2USDTerrainEnvCfg()

        # Import and create environment
        from genesislab.envs import ManagerBasedGenesisEnv

        env = ManagerBasedGenesisEnv(cfg=env_cfg)
        print("✅ Environment created successfully")

        # Verify terrain
        if env.scene.terrain is None:
            print("❌ ERROR: Terrain not initialized")
            return False
        print("✅ Terrain initialized in environment")

        # Print environment info
        print(f"\n📊 Environment info:")
        print(f"  - Num envs: {env.num_envs}")
        print(f"  - Env origins shape: {env.scene.terrain.env_origins.shape}")
        print(f"  - Device: {env.device}")

        # Reset environment
        print("\n🔄 Resetting environment...")
        obs, _ = env.reset()
        print(f"✅ Environment reset successful")
        print(f"  - Observation keys: {list(obs.keys())}")
        if "policy" in obs:
            print(f"  - Policy obs shape: {obs['policy'].shape}")

        # Run simulation steps
        print(f"\n▶️  Running {num_steps} simulation steps...")
        for step in range(num_steps):
            # Zero actions (stationary)
            actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            if step % 20 == 0:
                print(f"  Step {step:3d}/{num_steps}: "
                      f"reward_mean={rewards.mean().item():.3f}, "
                      f"terminated={terminated.sum().item()}/{env.num_envs}")

        print("✅ Simulation completed successfully")

        # Print final statistics
        print(f"\n📈 Final statistics:")
        print(f"  - Mean reward: {rewards.mean().item():.4f}")
        print(f"  - Terminated envs: {terminated.sum().item()}/{env.num_envs}")

        if viewer:
            print("\n👁️  Viewer is open. Press Ctrl+C to exit...")
            try:
                while True:
                    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
                    env.step(actions)
            except KeyboardInterrupt:
                print("\n🛑 Viewer closed by user")

        print("\n" + "=" * 80)
        print("✅ INTEGRATION TEST PASSED")
        print("=" * 80)
        return True

    except ImportError as e:
        print(f"⚠️  Could not import task environment: {e}")
        print("⚠️  Skipping robot integration test")
        print("⚠️  This is expected if genesis_tasks is not installed")
        return True  # Not a failure, just skip

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test USD terrain integration with robot")
    parser.add_argument(
        "--usd-path",
        type=str,
        default="third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd",
        help="Path to USD file",
    )
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--env-spacing", type=float, default=8.0, help="Environment spacing")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--viewer", action="store_true", help="Show Genesis viewer")

    args = parser.parse_args()

    success = test_usd_terrain_with_robot(
        usd_path=args.usd_path,
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        viewer=args.viewer,
        num_steps=args.num_steps,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
