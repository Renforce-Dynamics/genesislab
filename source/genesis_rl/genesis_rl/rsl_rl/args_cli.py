"""Common CLI argument helpers for GenesisLab + RSL-RL tools.

This module mirrors the IsaacLab ``isaaclab_rl.rsl_rl`` pattern where
training / evaluation scripts share a common set of command-line options.
"""

from __future__ import annotations

import argparse


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common RL arguments to the given parser.

    This function is intentionally generic so it can be re-used across:

    - Training scripts (e.g. ``train.py``)
    - Evaluation / play scripts
    - Distillation / imitation learning utilities
    """
    # Environment / task
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help=(
            "Gymnasium environment ID, e.g. 'Genesis-Velocity-Flat-Go2-v0'. "
            "If provided, env/rsl_rl cfg entry-points are read from the registered spec kwargs."
        ),
    )
    parser.add_argument(
        "--env-cfg-entry",
        type=str,
        required=False,
        help=(
            "Entry point to a ManagerBasedRlEnvCfg, e.g. "
            "'genesis_tasks.locomotion.velocity.velocity_env_cfg:VelocityEnvCfg'."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (if supported by the Scene config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # Training config
    parser.add_argument(
        "--train-cfg",
        type=str,
        required=False,
        help="Path to an rsl_rl OnPolicyRunner configuration in YAML format.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/rsl_rl",
        help="Base directory for logs and checkpoints.",
    )

    # Devices
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for training (e.g. 'cuda:0' or 'cpu').",
    )

    # Training iterations (used by training / evaluation scripts)
    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Number of learning iterations (override train-cfg if provided).",
    )


def add_play_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for play/inference scripts.

    This builds on top of :func:`add_common_args` and extends it with:

    - ``--checkpoint``: path to the trained model weights.
    - ``--num-episodes`` / ``--max-steps``: rollout length controls.
    """
    add_common_args(parser)

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint saved by OnPolicyRunner.save().",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to play.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of environment steps per episode (defaults to env horizon).",
    )


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for quantitative evaluation scripts.

    This is similar to :func:`add_play_args` but with a higher default number
    of episodes and without changing the semantics of common arguments.
    """
    add_play_args(parser)
    # Bump default number of episodes for evaluation if user didn't override it.
    for action in parser._actions:
        if action.dest == "num_episodes" and action.default == 10:
            action.default = 50
            action.help = "Number of episodes to evaluate (default: 50)."
            break
