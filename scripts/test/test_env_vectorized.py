"""Vectorization test for GenesisLab environments.

This script validates that environments work correctly with multiple
parallel environments, including batch operations and mid-rollout resets.

Usage:
    python scripts/test_env_vectorized.py
"""

import sys
import time
import argparse
from pathlib import Path

import genesis as gs
import torch
import gymnasium as gym

from genesislab.cli import add_viewer_args
from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from genesis_rl.rsl_rl.gym_utils import resolve_env_cfg_entry_point

from genesis_tasks import locomotion
import genesis_tasks.locomotion.velocity.robots.go2

def _load_env_cfg(entry_point: str) -> ManagerBasedRlEnvCfg:
    """Load a ``ManagerBasedRlEnvCfg`` from a module entry point.

    Args:
        entry_point: String of the form ``"module.path:ClassName"``.
    """
    module_name, class_name = entry_point.split(":")
    module = __import__(module_name, fromlist=[class_name])
    cfg_cls = getattr(module, class_name)
    return cfg_cls()


def test_vectorized(
    env_id: str = "Genesis-Velocity-Flat-Go2-v0",
    window: bool = True,
    render: bool = False,
    video: str | None = None,
) -> bool:
    """Run vectorized environment test.

    Args:
        env_id: Gym environment ID to test (e.g., "Genesis-Velocity-Flat-Go2-v0").
        window: Whether to show the Genesis viewer window.
        render: Whether to slow down stepping for human-viewable rendering.
        video: Optional path to save a rendered video (currently not implemented).
    """
    print("=" * 60)
    print("GenesisLab Vectorization Test")
    print("=" * 60)
    print(f"Environment ID: {env_id}")

    # Initialize Genesis
    backend_str = "cuda" if torch.cuda.is_available() else "cpu"
    backend = gs.gpu if backend_str == "cuda" else gs.cpu
    gs.init(backend=backend)

    # Load environment config from gym registry
    try:
        env_cfg_entry_point = resolve_env_cfg_entry_point(env_id)
        cfg = _load_env_cfg(env_cfg_entry_point)
        print(f"✓ Loaded config from: {env_cfg_entry_point}")
    except Exception as e:
        print(f"✗ Failed to load config from env_id '{env_id}': {e}")
        import traceback
        traceback.print_exc()
        return False
    # If we want to look at the motion, default to a single env; otherwise keep a large batch.
    cfg.scene.num_envs = 1 if (window or render or video is not None) else 4096
    cfg.scene.backend = backend_str
    # Control whether the viewer window is shown.
    cfg.scene.viewer = bool(window)
    # If a video path is provided, let the binding attach a camera and start recording.
    if video is not None:
        cfg.scene.record_video_path = str(Path(video))

    print(f"\nCreating environment with {cfg.scene.num_envs} env(s)...")
    try:
        env = ManagerBasedRlEnv(cfg)
        print(f"✓ Environment created successfully")
        print(f"  - Device: {env.device}")
        print(f"  - Num envs: {env.num_envs}")
        print(f"  - Action dim: {env.action_manager.total_action_dim}")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    scene = env._binding.scene
    # Test reset
    print("\nTesting reset...")
    try:
        obs, info = env.reset()
        print(f"✓ Reset successful")
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                expected_shape = (env.num_envs,) + value.shape[1:]
                if value.shape[0] == env.num_envs:
                    print(f"  - {key}: shape {value.shape} ✓ (correct batch size)")
                else:
                    print(f"  - {key}: shape {value.shape} ✗ (expected batch size {env.num_envs})")
                    return False
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test vectorized stepping
    print("\nTesting vectorized stepping (200 steps)...")
    try:
        total_reward = torch.zeros(env.num_envs, device=env.device)
        reset_count = 0

        for step in range(int(220)):
            # Random actions
            action = torch.randn(
                (env.num_envs, env.action_manager.total_action_dim),
                device=env.device
            )
            action = torch.clamp(action, -1.0, 1.0)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

            # Optional slow stepping for visual inspection
            if render or window:
                time.sleep(1.0 / 60.0)

            # Verify batch shapes
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    if value.shape[0] != env.num_envs:
                        print(f"✗ Observation {key} has incorrect batch size: {value.shape[0]} != {env.num_envs}")
                        return False

            if reward.shape[0] != env.num_envs:
                print(f"✗ Reward has incorrect batch size: {reward.shape[0]} != {env.num_envs}")
                return False

            if terminated.shape[0] != env.num_envs:
                print(f"✗ Terminated has incorrect batch size: {terminated.shape[0]} != {env.num_envs}")
                return False

            total_reward += reward

            # Reset terminated environments
            if terminated.any() or truncated.any():
                reset_envs = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_envs) > 0:
                    reset_count += 1
                    obs, info = env.reset(env_ids=reset_envs)

            if (step + 1) % 50 == 0:
                avg_reward = total_reward.mean().item()
                print(f"  Step {step + 1}/200: avg reward = {avg_reward:.4f}, resets = {reset_count}")

        print(f"✓ Vectorized stepping successful")
        print(f"  - Total steps: 200")
        print(f"  - Final avg reward: {total_reward.mean().item():.4f}")
        print(f"  - Total resets: {reset_count}")
        print(f"  - All batch shapes correct ✓")
        # Finalize any active recordings started in GenesisBinding.
        if video is not None:
            scene.stop_recording()
    except Exception as e:
        print(f"✗ Vectorized stepping failed: {e}")
        import traceback
        traceback.print_exc()
        if video is not None:
            try:
                scene.stop_recording()
            except Exception:
                pass
        return False

    print("\n" + "=" * 60)
    print("✓ All vectorization tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        default="Genesis-Velocity-Flat-Go2-v0",
        help="Gym environment ID to test (default: Genesis-Velocity-Flat-Go2-v0)",
    )
    add_viewer_args(parser)
    args = parser.parse_args()

    success = test_vectorized(
        env_id=args.env_id,
        window=args.window,
        render=args.render,
        video=args.video,
    )
    sys.exit(0 if success else 1)
