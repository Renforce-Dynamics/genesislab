"""Play a trained GenesisLab + RSL-RL policy.

This script loads a trained policy checkpoint and rolls out a few episodes
in the specified GenesisLab environment, similar in spirit to IsaacLab's
RL play utilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml
import argparse

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import gymnasium as gym

from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from genesislab.engine.visualize import stop_video_recorder
from genesis_rl.rsl_rl import GenesisRslRlVecEnv
from genesis_rl.rsl_rl.gym_utils import resolve_env_cfg_entry_point
from genesis_rl.rsl_rl.args_cli import add_play_args

from genesis_tasks.locomotion import *

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
    # Viewer flag: allow forcing the Genesis window on for interactive play.
    if getattr(args, "window", False) and hasattr(cfg, "scene") and hasattr(cfg.scene, "viewer"):
        setattr(cfg.scene, "viewer", True)


def _load_train_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_paths_from_checkpoint(args: argparse.Namespace) -> None:
    """Infer env-id, train-cfg and log-dir from the checkpoint path when possible.

    This allows a minimal CLI where the user only provides ``--checkpoint`` and
    everything else is resolved relative to the run directory, i.e.:

    - ``train.yaml`` is expected at ``{run_dir}/params/train.yaml``
    - ``env_id`` is read from ``train.yaml['env_id']`` if not provided
    - ``log_dir`` defaults to the checkpoint run directory if not overridden
    """
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    run_dir = ckpt_path.parent
    params_dir = run_dir / "params"

    # Infer train-cfg path if not provided.
    if not getattr(args, "train_cfg", None):
        candidate = params_dir / "train.yaml"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Could not infer train config for checkpoint '{ckpt_path}'.\n"
                f"Expected to find: {candidate}"
            )
        args.train_cfg = str(candidate)

    # Infer log-dir if user did not explicitly override it (assume default).
    # Default from args_cli is ``runs/rsl_rl``; in that case we prefer the run dir.
    if getattr(args, "log_dir", None) in (None, "runs/rsl_rl"):
        args.log_dir = str(run_dir)

    # Infer env-id from train.yaml if missing.
    if getattr(args, "env_id", None) is None and args.train_cfg is not None:
        train_cfg = _load_train_cfg(args.train_cfg)
        env_id = train_cfg.get("env_id", None)
        if env_id is None:
            raise ValueError(
                "No '--env-id' provided and the inferred train config does not "
                "contain an 'env_id' field.\n"
                f"Train config path: {args.train_cfg}\n"
                "Please either re-train with a newer train script (which stores 'env_id' "
                "into train.yaml) or pass '--env-id' explicitly."
            )
        args.env_id = env_id


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained GenesisLab task with RSL-RL.")
    add_play_args(parser)
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="If set, record a video of the rollout to LOG_DIR/videos/play.mp4.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=50,
        help="FPS for the recorded video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # If possible, infer env-id, train-cfg and log-dir from the checkpoint path
    # so that users can simply provide --checkpoint.
    _infer_paths_from_checkpoint(args)

    # Prepare log directory early so we can derive video paths from it.
    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    video_path: Path | None = None
    if args.video:
        video_dir = Path(log_dir) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "play.mp4"

    # ------------------------------------------------------------------ #
    # Environment construction
    # ------------------------------------------------------------------ #
    if args.env_id is not None:
        env_cfg_entry_point = resolve_env_cfg_entry_point(args.env_id)
        env_cfg = _load_env_cfg(env_cfg_entry_point)
        _apply_cli_overrides(env_cfg, args)
        if video_path is not None and hasattr(env_cfg, "scene"):
            setattr(env_cfg.scene, "record_video_path", str(video_path))
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    else:
        if not args.env_cfg_entry:
            raise ValueError("Either '--env-id' or '--env-cfg-entry' must be provided.")
        env_cfg = _load_env_cfg(args.env_cfg_entry)
        _apply_cli_overrides(env_cfg, args)
        if video_path is not None and hasattr(env_cfg, "scene"):
            setattr(env_cfg.scene, "record_video_path", str(video_path))
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

    vec_env = GenesisRslRlVecEnv(env)

    # ------------------------------------------------------------------ #
    # Runner and checkpoint loading
    # ------------------------------------------------------------------ #
    train_cfg = _load_train_cfg(args.train_cfg)

    # For play we don't use num_learning_iterations, but the runner expects it
    # to be present for logging. If missing, set a dummy value.
    train_cfg.setdefault("num_learning_iterations", 1)

    runner = OnPolicyRunner(vec_env, train_cfg=train_cfg, log_dir=log_dir, device=args.device)
    print(f"[GenesisLab][rsl_rl] Loading checkpoint from: {args.checkpoint}")
    runner.load(args.checkpoint, map_location=args.device)

    # ------------------------------------------------------------------ #
    # Rollout loop
    # ------------------------------------------------------------------ #
    num_envs = vec_env.num_envs
    max_episode_length = int(vec_env.max_episode_length) if vec_env.max_episode_length is not None else None

    if args.max_steps is not None:
        horizon = args.max_steps
    elif max_episode_length is not None:
        horizon = max_episode_length
    else:
        horizon = 1000  # sensible fallback

    print(
        f"[GenesisLab][rsl_rl] Playing {args.num_episodes} episodes "
        f"with horizon {horizon} steps (num_envs={num_envs})."
    )

    import tqdm

    try:
        for ep in range(args.num_episodes):
            # Reset underlying Genesis env and sync observations.
            env.reset(seed=args.seed)
            obs = vec_env.get_observations()

            episode_reward = torch.zeros(num_envs, device=args.device)
            pbar = tqdm.tqdm(range(horizon))

            for step in pbar:
                with torch.no_grad():
                    actions = runner.alg.act(obs)
                obs, rewards, dones, extras = vec_env.step(actions.to(vec_env.device))

                episode_reward += rewards.to(args.device)

            mean_reward = episode_reward.mean().item()
            print(f"[GenesisLab][rsl_rl][Play] Episode {ep + 1}/{args.num_episodes} - mean reward: {mean_reward:.3f}")
    finally:
        # Ensure any active recording is stopped cleanly.
        if args.video and hasattr(env, "_binding"):
            stop_video_recorder(env._binding.scene)


if __name__ == "__main__":
    # Initialize Genesis engine before creating any environments.
    gs.init(logging_level="WARNING")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    main()

