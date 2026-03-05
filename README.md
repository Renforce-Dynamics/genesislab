# GenesisLab: Fast and Simple !

<p align="center">
  <img src="./.docs/assets/intro/genesislab_banner.png" alt="Multi-Modal Whole-Body Control" width="100%" />
</p>

**GenesisLab** is a lightweight **reinforcement learning task suite** built on top of the **Genesis physics engine**.

It provides a minimal yet practical framework for:

- building RL environments
- running vectorized simulations
- debugging observations, rewards, and physics behaviors

The repository is intended as a **fast experimentation playground** for robotics RL on Genesis.

---

## ✨ Features

- **RL Task Library**
  - Ready-to-use environments (e.g. **Unitree Go2 velocity tracking**)
  - Consistent step/reset interfaces

- **Vectorized Simulation**
  - Batched environments for efficient RL rollouts

- **Debugging Utilities**
  - Engine smoke tests
  - Environment sanity checks
  - Random rollout scripts

- **Minimal & Hackable**
  - Small codebase designed for rapid task iteration

---

# Installation

GenesisLab depends on the **Genesis physics engine**.

1. Create Environment

```bash
conda create -n genesislab python=3.10
conda activate genesislab
```

2. Install Genesis

```bash
pip install genesis-world
```

3. Install GenesisLab

From the repository root:

```bash
bash ./scripts/setup/setup_ext.sh
```

This installs the `genesislab` Python package in **editable mode**.

---

# Quick Start

### Engine Smoke Test

Verify Genesis backend and Python bindings:

```bash
python scripts/test/test_engine.py --backend cpu --num-envs 4
```

### RL Environment Tests

```bash
# single environment
python scripts/test/test_env.py

# vectorized environments
python scripts/test/test_env_vectorized.py

# random rollout stress test
python scripts/test/test_random_rollout.py
```

These scripts validate:

- Genesis bindings
- RL environment stepping
- reward and observation pipelines
- vectorized rollouts

---

# Repository Structure

<details>
<summary>Click to expand</summary>

```
genesislab/
├── source/genesislab/
│   ├── cli/                  # CLI utilities
│   ├── engine/               # Genesis bindings
│   ├── tasks/                # RL task definitions
│   └── envs/                 # environment wrappers
│
├── scripts/
│   ├── setup/
│   │   └── setup_ext.sh      # installation script
│   └── test/
│       ├── test_engine.py
│       ├── test_env.py
│       ├── test_env_vectorized.py
│       └── test_random_rollout.py
│
├── README.md
└── pyproject.toml
```

</details>

---

# Development

<details>
<summary>Development notes</summary>

GenesisLab is designed to be easy to extend.

Typical workflow for adding a new RL task:

1. Duplicate an existing task as a template
2. Implement observation / reward logic
3. Add a minimal test script in `scripts/test`
4. Run sanity checks before training

Recommended tests:

```
test_env.py
test_env_vectorized.py
test_random_rollout.py
```

</details>

---

# License

See the `LICENSE` file for details.

If you use GenesisLab in research or open-source projects,  
please consider citing or linking this repository.