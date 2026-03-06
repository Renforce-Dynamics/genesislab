"""Train a GenesisLab task with RSL-RL (PPO runner).

This script is intentionally close in spirit to IsaacLab's
``scripts/reinforcement_learning/rsl_rl/train.py``, but wired to
GenesisLab's manager-based RL environments.
"""

from __future__ import annotations

import os
from datetime import datetime
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

import genesis_tasks.locomotion

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
            train_cfg = cfg_obj().to_dict()  # type: ignore[assignment]
        elif isinstance(cfg_obj, dict):
            train_cfg = dict(cfg_obj)
        else:
            raise TypeError(
                "Resolved rsl_rl config object is neither a dict nor has 'to_dict()'."
            )

    # Optional override of num_learning_iterations.
    if args.num_iters is not None:
        train_cfg["max_iterations"] = args.num_iters

    # Persist env-id into train config so that play/eval scripts can infer it
    # from the saved checkpoint directory without requiring it on the CLI.
    if args.env_id is not None:
        # Don't overwrite if the config already specifies an env_id.
        train_cfg.setdefault("env_id", args.env_id)

    # ------------------------------------------------------------------ #
    # Logging directory (aligned with IsaacLab-style structure)
    # ------------------------------------------------------------------ #
    # Base root for all RSL-RL runs (default: "runs/rsl_rl")
    base_root = os.path.abspath(args.log_dir)

    # Experiment name: prefer config-provided, otherwise derive from env-id.
    experiment_name = train_cfg.get("experiment_name")
    if not experiment_name:
        experiment_name = args.env_id or "default_experiment"

    log_root_path = os.path.join(base_root, str(experiment_name))
    print(f"[GenesisLab][rsl_rl] Logging experiment in directory: {log_root_path}")

    # Per-run directory: {timestamp}_{optional_run_name}
    run_name = train_cfg.get("run_name")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[GenesisLab][rsl_rl] Exact run name requested from config: {log_dir}")
    if run_name:
        log_dir += f"_{run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Propagate log_dir into env config if it exposes such a field.
    if hasattr(env_cfg, "log_dir"):
        setattr(env_cfg, "log_dir", log_dir)

    os.makedirs(log_dir, exist_ok=True)
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    env_cfg_dict = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else None  # type: ignore[assignment]

    if env_cfg_dict is not None:
        with open(os.path.join(params_dir, "env.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(env_cfg_dict, f, sort_keys=False)
    # Training / agent config.
    with open(os.path.join(params_dir, "train.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False)

    # ------------------------------------------------------------------ #
    # Runner construction and main training loop
    # ------------------------------------------------------------------ #
    runner = OnPolicyRunner(vec_env, train_cfg=train_cfg, log_dir=log_dir, device=args.device)

    max_iterations = int(train_cfg.get("max_iterations"))

    print(f"[GenesisLab][rsl_rl] Starting training for {max_iterations} iterations...")
    runner.learn(num_learning_iterations=max_iterations)
    print("[GenesisLab][rsl_rl] Training finished.")


if __name__ == "__main__":
    # Initialize Genesis engine before creating any environments.
    gs.init(logging_level="WARNING")

    # Ensure TF32 is enabled for better performance when using CUDA.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    main()

