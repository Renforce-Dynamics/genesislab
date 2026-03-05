"""Runner configuration classes for RSL-RL (GenesisLab)."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from genesislab.utils.configclass import configclass

from .policy_cfg import RslRlPpoActorCriticCfg
from .algo_cfg import RslRlPpoAlgorithmCfg


@configclass
class RslRlOnPolicyRunnerCfg:
    """Base configuration for on-policy runners (e.g. PPO) in RSL-RL."""

    # Experiment / bookkeeping
    seed: int = 42
    """Random seed for the experiment."""

    device: str = "cuda:0"
    """RL device (e.g. 'cuda:0', 'cpu')."""

    num_steps_per_env: int = MISSING
    """Number of environment steps per update."""

    max_iterations: int = MISSING
    """Maximum number of optimization iterations."""

    # Observation group mapping; keys are algo obs sets,
    # values are lists of env observation groups.
    obs_groups: dict[str, list[str]] = MISSING

    # Action clipping (optional)
    clip_actions: float | None = None

    # Logging / saving
    save_interval: int = MISSING
    """Number of iterations between checkpoints."""

    experiment_name: str = MISSING
    """Experiment name used for log directory naming."""

    run_name: str = ""
    """Optional run-name suffix for log directories."""

    # Logger backends (kept simple; can be extended as needed)
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = "genesislab"

    # Checkpoint loading / resuming
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    # Nested configs
    policy: RslRlPpoActorCriticCfg = MISSING
    algorithm: RslRlPpoAlgorithmCfg = MISSING


__all__ = ["RslRlOnPolicyRunnerCfg"]

