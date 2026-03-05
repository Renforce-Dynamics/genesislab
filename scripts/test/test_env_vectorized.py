"""Vectorization test for GenesisLab environments.

This script validates that environments work correctly with multiple
parallel environments, including batch operations and mid-rollout resets.

Usage:
    python scripts/test_env_vectorized.py
"""

import sys
from pathlib import Path

import genesis as gs
import torch

from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv
from genesis_tasks.locomotion.velocity.robots.go2.go2_flat_env_cfg import Go2VelocityEnvCfg


def test_vectorized():
    """Run vectorized environment test."""
    print("=" * 60)
    print("GenesisLab Vectorization Test")
    print("=" * 60)

    # Initialize Genesis
    backend_str = "cuda" if torch.cuda.is_available() else "cpu"
    backend = gs.gpu if backend_str == "cuda" else gs.cpu
    gs.init(backend=backend)

    # Create config with multiple environments
    cfg = Go2VelocityEnvCfg()
    cfg.scene.num_envs = 4096  # Use 64 environments
    cfg.scene.backend = backend_str

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

        for step in range(200):
            # Random actions
            action = torch.randn(
                (env.num_envs, env.action_manager.total_action_dim),
                device=env.device
            )
            action = torch.clamp(action, -1.0, 1.0)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

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
    except Exception as e:
        print(f"✗ Vectorized stepping failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All vectorization tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_vectorized()
    sys.exit(0 if success else 1)
