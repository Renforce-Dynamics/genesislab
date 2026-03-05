"""Train a GenesisLab task with RSL-RL (PPO runner).

This script is intentionally close in spirit to IsaacLab's
``scripts/reinforcement_learning/rsl_rl/train.py``, but wired to
GenesisLab's manager-based RL environments.
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
from genesis_rl.rsl_rl.args_cli import add_common_args
from genesis_rl.rsl_rl.gym_utils import resolve_env_cfg_entry_point, resolve_rsl_rl_cfg_object

from genesis_tasks.locomotion.velocity.robots.go2 import Go2FlatVelocityEnvCfg

def _load_env_cfg(entry_point: str) -> ManagerBasedRlEnvCfg:
    """Load a ``ManagerBasedRlEnvCfg`` from a module entry point.

    Args:
        entry_point: String of the form ``"module.path:ClassName"``.
    """
    module_name, class_name = entry_point.split(":")
    module = __import__(module_name, fromlist=[class_name])
    cfg_cls = getattr(module, class_name)
    return cfg_cls()


def _apply_cli_overrides(cfg: ManagerBasedRlEnvCfg, args: argparse.Namespace) -> None:
    """Apply a few common command-line overrides to the env config."""
    if hasattr(cfg, "seed") and args.seed is not None:
        cfg.seed = args.seed

    # Try to override number of envs if the scene exposes such a field.
    if args.num_envs is not None and hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
        setattr(cfg.scene, "num_envs", args.num_envs)


def _load_train_cfg(path: str) -> dict[str, Any]:
    """Load an rsl_rl runner configuration from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GenesisLab task with RSL-RL (PPO).")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Environment construction
    # ------------------------------------------------------------------ #
    if args.env_id is not None:
        # Use Gym registry only to read env_cfg entry-point.
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
    # Training configuration and runner
    # ------------------------------------------------------------------ #
    if args.train_cfg:
        train_cfg = _load_train_cfg(args.train_cfg)
    else:
        # Optionally load from a registered rsl_rl cfg entry-point if provided.
        cfg_obj = resolve_rsl_rl_cfg_object(args.env_id) if args.env_id is not None else None

        if cfg_obj is None:
            raise ValueError(
                "No '--train-cfg' provided and gym spec.kwargs has no 'rsl_rl_cfg_entry_point'."
            )

        # Support either a dict-like object or an object with ``to_dict()``.
        if hasattr(cfg_obj, "to_dict"):
            train_cfg = cfg_obj.to_dict()  # type: ignore[assignment]
        elif isinstance(cfg_obj, dict):
            train_cfg = dict(cfg_obj)
        else:
            raise TypeError(
                "Resolved rsl_rl config object is neither a dict nor has 'to_dict()'."
            )

    # Optional override of num_learning_iterations.
    if args.num_iters is not None:
        train_cfg["num_learning_iterations"] = args.num_iters

    # Prepare log directory
    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(vec_env, train_cfg=train_cfg, log_dir=log_dir, device=args.device)

    num_learning_iterations = int(train_cfg.get("num_learning_iterations", 0))
    if num_learning_iterations <= 0 and args.num_iters is None:
        raise ValueError(
            "Number of learning iterations must be specified either in 'train-cfg' "
            "under 'num_learning_iterations' or via '--num-iters'."
        )
    if args.num_iters is not None:
        num_learning_iterations = args.num_iters

    print(f"[GenesisLab][rsl_rl] Starting training for {num_learning_iterations} iterations...")
    runner.learn(num_learning_iterations=num_learning_iterations)
    print("[GenesisLab][rsl_rl] Training finished.")


if __name__ == "__main__":
    # Initialize Genesis engine before creating any environments.
    gs.init()

    # Ensure TF32 is enabled for better performance when using CUDA.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    main()

