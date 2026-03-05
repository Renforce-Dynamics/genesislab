GenesisLab Third-Party Dependencies
===================================

This directory is used to centrally manage third-party repositories used by GenesisLab (e.g., RL libraries).

- **Config file**: `third_party_repos.yaml`
- **Setup script**: `setup_third_party.sh`

### YAML configuration format

`third_party_repos.yaml` uses a simple YAML list to describe the repositories to clone and whether they should be installed via `pip install -e`:

```yaml
repos:
  - name: rsl_rl            # Only for identification; the script does not depend on this field
    url: https://github.com/leggedrobotics/rsl_rl.git
    path: ./third_party/rsl_rl
    editable: true          # true: pip install -e path; false: only clone, do not install
```

- **url**: Git repository URL.
- **path**: Target path inside this project (relative path is fine, e.g. `./third_party/rsl_rl`).
- **editable**:
  - `true`  → after cloning, the script will run `pip install -e path`
  - `false` → only clone, no Python install

You can add more third-party repositories by appending entries in the same format.

### Usage

From the `genesislab` directory, run:

```bash
bash ./third_party/setup_third_party.sh
```

The script will:

1. Read `third_party_repos.yaml`.
2. Clone each missing repository (skip if it already exists).
3. For repositories with `editable: true`, run `pip install -e` on their paths.

You can also specify an alternative YAML file via environment variable:

```bash
THIRD_PARTY_YAML=/path/to/your_repos.yaml bash ./third_party/setup_third_party.sh
```
