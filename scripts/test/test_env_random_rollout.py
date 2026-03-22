"""Smoke test: build a registered task env and step with zero actions.

Mirrors ``scripts/reinforcement_learning/rsl_rl/play.py`` setup (Genesis init,
task imports for ``gym.register``) but skips training — only reset + a short
zero-action rollout.
"""

from __future__ import annotations

import argparse

import genesis as gs
import torch

import genesis_tasks.locomotion  # noqa: F401
import genesis_tasks.imitation.tracking  # noqa: F401
from genesis_rl.rsl_rl.gym_utils import resolve_env_cfg_entry_point
from genesis_rl.rsl_rl.utils.env_cfg import load_env_cfg
from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


def run_smoke(
    env_id: str,
    num_envs: int | None,
    num_steps: int,
    backend: str,
) -> bool:
    entry = resolve_env_cfg_entry_point(env_id)
    cfg = load_env_cfg(entry)
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    cfg.scene.backend = backend

    env = ManagerBasedRlEnv(cfg=cfg, device=backend)
    try:
        obs, _ = env.reset()
        adim = env.action_manager.total_action_dim
        z = torch.zeros(env.num_envs, adim, device=env.device)
        for _ in range(num_steps):
            _, _, term, trunc, _ = env.step(z)
            if term.any() or trunc.any():
                reset_ids = (term | trunc).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    env.reset(env_ids=reset_ids)
        return True
    finally:
        del env


def main() -> int:
    p = argparse.ArgumentParser(description="Registered env smoke test (zero actions).")
    p.add_argument("--env-id", type=str, default="Genesis-Velocity-Flat-Go2-v0")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = p.parse_args()
    try:
        backend = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        gs.init(backend=gs.gpu if backend == "cuda" else gs.cpu, logging_level="WARNING")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False

        ok = run_smoke(args.env_id, args.num_envs, args.num_steps, backend)
        print(f"OK: {args.env_id} stepped {args.num_steps} times (zero actions).")
        return 0 if ok else 1
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
