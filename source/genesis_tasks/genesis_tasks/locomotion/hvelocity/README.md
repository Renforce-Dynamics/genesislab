# Human velocity (`hvelocity`)

`hvelocity` (**human velocity**) is the biped counterpart of the generic `locomotion/velocity` stack: the same *manager-based* RL layout (scene, commands, observations, rewards, terminations, events, curriculum), but MDP code and defaults are **humanoid-specific** and live only under this package.

## Why a separate tree?

- **Morphology**: arms, waist, feet contact, gait, and undesired-contact masks are expressed for humanoids (G1 first), not quadruped defaults.
- **No cross-import**: Task code does **not** import `genesis_tasks.locomotion.velocity` so quadruped and human pipelines can evolve independently.
- **Reference**: Behaviour and hyper-parameters are migrated from  
  `.references/soccerLab/.../locomotion/velocity` (SoccerLab / IsaacLab-style human locomotion), then adapted to GenesisLab APIs.

## Layout (mirrors `locomotion/velocity`)

| Path | Role |
|------|------|
| `components.py` | Scene, commands, observations, rewards, terminations, events, curriculum configs |
| `base_hvelocity_env_cfg.py` | `BaseHumanVelocityEnvCfg` — wires managers and episode settings |
| `velocity_env_cfg.py` | `HumanVelocityEnvCfg` / `HumanVelocityPlayEnvCfg` |
| `mdp/` | Commands, rewards, observations, terminations, events, curriculums (human-oriented) |
| `g1/` | Unitree G1 task entry points and PPO config |

## Design choices

1. **Commands**: `UniformLevelVelocityCommandCfg` exposes both `ranges` (initial sampling) and `limit_ranges` (ceiling). Curriculum `lin_vel_cmd_levels` linearly expands the active `ranges` toward `limit_ranges` over training steps. Play configs set `ranges = limit_ranges` so evaluation uses the full envelope.
2. **Tracking rewards**: Linear/angular velocity tracking uses exponential kernels on body-frame errors (same spirit as the reference `track_lin_vel_xy_yaw_frame_exp` naming).
3. **Human shaping**: Default posture penalties (arms / waist / selected leg joints), gait / slide / clearance on ankle roll links, mechanical power penalty, height and orientation terms — aligned with the migrated SoccerLab weights where applicable.
4. **Flat G1 task**: `g1/flat_env_cfg.py` switches terrain to `plane` and disables terrain-level curriculum (no generator), keeping only command curriculum.

## Gym registration

- `Genesis-HVelocity-G1-v0` → `g1.flat_env_cfg:RobotEnvCfg` / `RobotPlayEnvCfg`.
