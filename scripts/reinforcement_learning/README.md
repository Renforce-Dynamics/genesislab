GenesisLab Reinforcement Learning Scripts
=========================================

This directory contains reinforcement learning (RL) entry-points for training
GenesisLab tasks using the `rsl_rl` library, following the structure of
IsaacLab's `scripts/reinforcement_learning/rsl_rl` utilities.

The first supported workflow is:

- **Trainer**: `rsl_rl.OnPolicyRunner` (PPO)
- **Env base**: `genesislab.envs.ManagerBasedRlEnv`
- **Wrapper**: small adapter that exposes a `rsl_rl.env.VecEnv`-compatible API.

### Layout

```text
genesislab/scripts/reinforcement_learning/
├── README.md
└── rsl_rl/
    ├── __init__.py
    ├── env_wrappers.py      # Genesis → rsl_rl VecEnv adapter
    └── train.py             # Main training script
```

### Quick start

Example (velocity locomotion task, using a config entry point):

```bash
cd genesislab

python scripts/reinforcement_learning/rsl_rl/train.py \
  --env-cfg-entry genesis_tasks.locomotion.velocity.velocity_env_cfg:VelocityEnvCfg \
  --train-cfg /path/to/rsl_rl_train_cfg.yaml \
  --device cuda:0 \
  --seed 42
```

- `--env-cfg-entry` points to a `ManagerBasedRlEnvCfg` subclass.
- `--train-cfg` is an `rsl_rl` runner configuration in YAML format.

You can extend this directory with additional trainers (e.g. distillation
or evaluation scripts) following the same pattern.

