"""Evaluate a trained GenesisLab + RSL-RL policy over multiple episodes.

This script runs a trained policy and reports aggregate statistics such as
mean / std episode return, similar in spirit to IsaacLab's RL evaluation
helpers.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import yaml
import argparse

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import gymnasium as gym

from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from genesis_rl.rsl_rl import GenesisRslRlVecEnv
from genesis_rl.rsl_rl.gym_utils import resolve_env_cfg_entry_point
from genesis_rl.rsl_rl.args_cli import add_eval_args


def _load_env_cfg(entry_point: str) -> ManagerBasedRlEnvCfg:
    module_name, class_name = entry_point.split(":")
    module = __import__(module_name, fromlist=[class_name])
    cfg_cls = getattr(module, class_name)
    return cfg_cls()


def _apply_cli_overrides(cfg: ManagerBasedRlEnvCfg, args: argparse.Namespace) -> None:
    if hasattr(cfg, "seed") and args.seed is not None:
        cfg.seed = args.seed
    if args.num_envs is not None and hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
        setattr(cfg.scene, "num_envs", args.num_envs)


def _load_train_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GenesisLab task with RSL-RL.")
    add_eval_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Environment construction
    # ------------------------------------------------------------------ #
    if args.env_id is not None:
        env_cfg_entry_point = resolve_env_cfg_entry_point(args.env_id)
        env_cfg = _load_env_cfg(env_cfg_entry_point)
        _apply_cli_overrides(env_cfg, args)
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    else:
        if not args.env_cfg_entry:
            raise ValueError("Either '--env-id' or '--env-cfg-entry' must be provided.")
        env_cfg = _load_env_cfg(args.env_cfg_entry)
        _apply_cli_overrides(env_cfg, args)
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

    vec_env = GenesisRslRlVecEnv(env)

    # ------------------------------------------------------------------ #
    # Runner and checkpoint loading
    # ------------------------------------------------------------------ #
    train_cfg = _load_train_cfg(args.train_cfg)
    train_cfg.setdefault("num_learning_iterations", 1)

    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(vec_env, train_cfg=train_cfg, log_dir=log_dir, device=args.device)
    print(f"[GenesisLab][rsl_rl][Eval] Loading checkpoint from: {args.checkpoint}")
    runner.load(args.checkpoint, map_location=args.device)

    num_envs = vec_env.num_envs
    max_episode_length = int(vec_env.max_episode_length) if vec_env.max_episode_length is not None else None

    if args.max_steps is not None:
        horizon = args.max_steps
    elif max_episode_length is not None:
        horizon = max_episode_length
    else:
        horizon = 1000

    print(
        f"[GenesisLab][rsl_rl][Eval] Evaluating {args.num_episodes} episodes "
        f"with horizon {horizon} steps (num_envs={num_envs})."
    )

    all_returns = []

    for ep in range(args.num_episodes):
        env.reset(seed=args.seed)
        obs = vec_env.get_observations()

        episode_reward = torch.zeros(num_envs, device=args.device)

        for step in range(horizon):
            with torch.no_grad():
                actions = runner.alg.act(obs)
            obs, rewards, dones, extras = vec_env.step(actions.to(vec_env.device))

            episode_reward += rewards.to(args.device)

            if dones.any():
                break

        mean_reward = episode_reward.mean().item()
        all_returns.append(mean_reward)
        print(
            f"[GenesisLab][rsl_rl][Eval] Episode {ep + 1}/{args.num_episodes} "
            f"- mean reward: {mean_reward:.3f}"
        )

    returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
    mean_return = returns_tensor.mean().item()
    std_return = returns_tensor.std(unbiased=False).item()

    print(
        f"[GenesisLab][rsl_rl][Eval] Completed {args.num_episodes} episodes.\n"
        f"  Mean return: {mean_return:.3f}\n"
        f"  Std  return: {std_return:.3f}"
    )


if __name__ == "__main__":
    # Initialize Genesis engine before creating any environments.
    gs.init()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    main()

