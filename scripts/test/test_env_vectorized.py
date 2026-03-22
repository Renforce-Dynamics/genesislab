"""Vectorized smoke test for registered GenesisLab environments.

Uses the Gymnasium registry (``env_cfg_entry_point``) like training/play scripts,
then builds :class:`~genesislab.envs.manager_based_rl_env.ManagerBasedRlEnv`
and steps with **zero** actions to validate batching, reset, and observations.

Usage:
    python scripts/test/test_env_vectorized.py
    python scripts/test/test_env_vectorized.py --env-id Genesis-Velocity-Flat-Go2-v0 --num-envs 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import genesis as gs
import torch

import genesis_tasks.locomotion  # noqa: F401
import genesis_tasks.imitation.tracking  # noqa: F401
from genesis_rl.rsl_rl.gym_utils import resolve_env_cfg_entry_point
from genesis_rl.rsl_rl.utils.env_cfg import load_env_cfg
from genesislab.cli import add_viewer_args
from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


def test_vectorized(
    env_id: str = "Genesis-Velocity-Flat-Go2-v0",
    num_envs: int | None = None,
    num_steps: int = 220,
    window: bool = False,
    render: bool = False,
    video: str | None = None,
    device: str | None = None,
) -> bool:
    backend = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gs.init(backend=gs.gpu if backend == "cuda" else gs.cpu, logging_level="WARNING")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    print("=" * 60)
    print("GenesisLab vectorized env smoke test (zero actions)")
    print("=" * 60)
    print(f"Environment ID: {env_id}")

    try:
        env_cfg_entry_point = resolve_env_cfg_entry_point(env_id)
        cfg = load_env_cfg(env_cfg_entry_point)
        print(f"✓ Loaded config from: {env_cfg_entry_point}")
    except Exception as e:
        print(f"✗ Failed to load config from env_id '{env_id}': {e}")
        import traceback

        traceback.print_exc()
        return False

    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    elif window or render or video is not None:
        cfg.scene.num_envs = 1
    else:
        cfg.scene.num_envs = 4096
    cfg.scene.backend = backend
    cfg.scene.viewer = bool(window)
    if video is not None:
        cfg.scene.record_video_path = str(Path(video))

    print(f"\nCreating environment with {cfg.scene.num_envs} env(s)...")
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=backend)
        print("✓ Environment created successfully")
        print(f"  - Device: {env.device}")
        print(f"  - Num envs: {env.num_envs}")
        print(f"  - Action dim: {env.action_manager.total_action_dim}")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    scene = env._scene
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

    print("\nTesting reset...")
    try:
        obs, _ = env.reset()
        print("✓ Reset successful")
        for key, value in obs.items():
            if isinstance(value, torch.Tensor) and value.shape[0] != env.num_envs:
                print(f"  - {key}: shape {value.shape} ✗ (expected batch size {env.num_envs})")
                return False
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape {value.shape} ✓")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print(f"\nTesting vectorized stepping ({num_steps} steps, zero actions)...")
    try:
        total_reward = torch.zeros(env.num_envs, device=env.device)
        reset_count = 0

        for step in range(num_steps):
            obs, reward, terminated, truncated, _ = env.step(zero)

            if render or window:
                time.sleep(1.0 / 60.0)

            for key, value in obs.items():
                if isinstance(value, torch.Tensor) and value.shape[0] != env.num_envs:
                    print(f"✗ Observation {key} has incorrect batch size: {value.shape[0]} != {env.num_envs}")
                    return False

            if reward.shape[0] != env.num_envs:
                print(f"✗ Reward has incorrect batch size: {reward.shape[0]} != {env.num_envs}")
                return False

            if terminated.shape[0] != env.num_envs:
                print(f"✗ Terminated has incorrect batch size: {terminated.shape[0]} != {env.num_envs}")
                return False

            total_reward += reward

            if terminated.any() or truncated.any():
                reset_envs = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_envs) > 0:
                    reset_count += 1
                    env.reset(env_ids=reset_envs)

            if (step + 1) % 50 == 0:
                avg_reward = total_reward.mean().item()
                print(f"  Step {step + 1}/{num_steps}: avg reward = {avg_reward:.4f}, resets = {reset_count}")

        print("✓ Vectorized stepping successful")
        print(f"  - Total steps: {num_steps}")
        print(f"  - Final avg reward: {total_reward.mean().item():.4f}")
        print(f"  - Total resets: {reset_count}")
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
        help="Gym environment ID registered by genesis_tasks",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Override cfg.scene.num_envs (default: 1 with viewer, else 4096).")
    parser.add_argument("--num-steps", type=int, default=220)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    add_viewer_args(parser)
    args = parser.parse_args()

    success = test_vectorized(
        env_id=args.env_id,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        window=args.window,
        render=args.render,
        video=args.video,
        device=args.device,
    )
    sys.exit(0 if success else 1)
