# Motion Tracking (Imitation) — Tutorial

Train a humanoid to follow a reference motion clip. The shipping example is
**Unitree G1 whole-body tracking** on flat terrain, registered as
`Genesis-Tracking-Flat-G1-v0`.

---

## Pipeline overview

```
mocap / AMASS  ──►  retargeted CSV  ──►  baked NPZ  ──►  MotionCommand  ──►  RL training
                     (GMR, external)     (fk_to_npz)       (loader)         (rsl_rl)
```

Three discrete steps, described below.

---

## 1. Retarget reference motion to the robot (GMR)

GenesisLab does **not** ship a retargeter. Motion retargeting — mapping a
source skeleton (SMPL / AMASS / LAFAN1 / raw mocap) onto the target robot's
URDF — is handled by an external tool.

Recommended: [**general-motion-retargeting (GMR)**](https://github.com/YanjieZe/GMR)
— produces per-frame root pose + joint angles aligned with the target robot's
DoF layout.

The expected handoff format is a CSV with one row per frame and columns:

```
base_pos_x, base_pos_y, base_pos_z,                    # 3
base_quat_x, base_quat_y, base_quat_z, base_quat_w,    # 4   (xyzw)
joint_pos_0, joint_pos_1, ..., joint_pos_{N-1}         # N
```

The base quaternion is **xyzw** (SciPy / IsaacLab convention); `fk_to_npz.py`
converts to wxyz internally. For G1 the joint order must match
`G1_JOINT_NAMES` in
[`source/.../tracking/scripts/fk_to_npz.py`][fk-to-npz] (29 DoFs, left-leg →
right-leg → waist → left-arm → right-arm). Any other order produces garbage.

A reference clip is bundled for smoke testing:
`data/datasets/dance1_subject1.csv` (6574 frames @ 30 Hz).

---

## 2. Bake the motion to NPZ (Genesis FK replay)

Run the retargeted joints through Genesis forward-kinematics, interpolate to
the training control frequency, and bake every link's world pose + velocity
into a single `.npz` file:

```bash
python source/genesis_tasks/genesis_tasks/imitation/tracking/scripts/fk_to_npz.py \
  --input-dir ./data/datasets \
  --output-dir ./data/datasets \
  --input-fps 30 \
  --output-fps 50
```

Every CSV in `--input-dir` is converted; add `--window` to watch the replay.

The script writes `<name>.npz` with these keys (T = number of output frames):

| Key              | Shape        | Notes                                   |
|------------------|--------------|-----------------------------------------|
| `fps`            | scalar       | Output FPS (50 by default)              |
| `joint_pos`      | `(T, 29)`    | Joint angles, robot DoF order           |
| `joint_vel`      | `(T, 29)`    | Finite-difference joint velocities      |
| `body_pos_w`     | `(T, 30, 3)` | World-frame positions of all 30 links   |
| `body_quat_w`    | `(T, 30, 4)` | Link orientations, **wxyz**             |
| `body_rot_w`     | `(T, 3)`     | Base Euler XYZ (convenience only)       |
| `body_lin_vel_w` | `(T, 30, 3)` | World-frame linear velocities           |
| `body_ang_vel_w` | `(T, 30, 3)` | World-frame angular velocities          |

The link order comes from the G1 USD asset and must not be reshuffled — the
tracking config indexes into `body_pos_w` by link name via the env's body
registry.

---

## 3. Point the config at the NPZ

The motion file path lives on the tracking command:

```python
# source/genesis_tasks/genesis_tasks/imitation/tracking/components.py
motion: mdp.MotionCommandCfg = mdp.MotionCommandCfg(
    asset_name="robot",
    motion_file="data/datasets/dance1_subject1.npz",   # ← change here
    anchor_body_name=MISSING,   # set per robot (e.g. "torso_link" for G1)
    body_names=MISSING,         # list of links to track (G1 tracks 14)
    ...
)
```

Robot-specific overrides live in
[`.../tracking/robots/g1/g1_tracking_env_cfg.py`][g1-cfg] — that's where the
G1 tracking task pins `anchor_body_name="torso_link"` and the 14-body
tracking list (pelvis, ankles, knees, torso, shoulders, elbows, wrists).

To train on a different clip, either:

- Point `motion_file` at your new `.npz`, or
- Subclass `G1FlatEnvCfg` and override `commands.motion.motion_file` in
  `__post_init__`.

The loader class is `MotionCommand` in
[`.../tracking/mdp/commands.py`][motion-cmd]; reward + observation terms live
alongside it in the same `mdp/` package.

---

## 4. Train

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --env-id Genesis-Tracking-Flat-G1-v0 \
  --num-envs 4096 \
  --num-iters 5000
```

Variants registered for G1:

| Env ID                                               | Notes                               |
|------------------------------------------------------|-------------------------------------|
| `Genesis-Tracking-Flat-G1-v0`                        | Full observations (with anchor pos + base lin vel) |
| `Genesis-Tracking-Flat-G1-Wo-State-Estimation-v0`    | Drops anchor pos + base lin vel from policy obs |
| `Genesis-Tracking-Flat-G1-Low-Freq-v0`               | Lower control rate (higher decimation) |

Play back a trained checkpoint with the standard `play.py`:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --env-id Genesis-Tracking-Flat-G1-v0 \
  --window \
  --num-envs 1 \
  --checkpoint <PATH_TO_CHECKPOINT>
```

---

## Retargeting to other humanoids

The tracking env itself is robot-agnostic — it just needs an NPZ with
matching joint count and the link names referenced in the robot-specific cfg.
To add a new humanoid:

1. Retarget your reference motion with GMR (or an equivalent) onto the new
   robot's URDF.
2. Copy `fk_to_npz.py`, swap `G1_FULL_ACT_CFG` + `G1_JOINT_NAMES` for the new
   robot's asset and DoF order.
3. Subclass `TrackingEnvCfg`, set `anchor_body_name` and `body_names` to
   valid link names for that robot, and register a new gym ID.

---

## Troubleshooting

**Robot collapses / wrong posture at reset.** Almost always a joint-order
mismatch between the CSV and `G1_JOINT_NAMES`. Log the first frame's
`joint_pos` and verify left-leg joints are in the first 6 slots.

**Base orientation looks mirrored / flipped.** The CSV base quaternion is
xyzw but the NPZ stores wxyz. If you bypass `fk_to_npz.py` and write NPZ
directly, remember to convert.

**Training diverges immediately.** Check `body_names` in the robot cfg —
every name must exist in the asset's link registry. A typo silently selects
index 0 (usually the base) and the reward signal becomes nonsense.

**"Motion file not found".** `motion_file` is resolved relative to the
working directory when the env is built, not relative to the cfg file. Run
training from the repo root, or pass an absolute path.

[fk-to-npz]: ../../source/genesis_tasks/genesis_tasks/imitation/tracking/scripts/fk_to_npz.py
[g1-cfg]: ../../source/genesis_tasks/genesis_tasks/imitation/tracking/robots/g1/g1_tracking_env_cfg.py
[motion-cmd]: ../../source/genesis_tasks/genesis_tasks/imitation/tracking/mdp/commands.py
