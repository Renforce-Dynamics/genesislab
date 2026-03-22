# MDP joint actions

This package implements two ways to turn policy outputs into joint motion in Genesis. Both are **joint-DOF only**: floating-base degrees of freedom are never written by these terms.

## Degrees of freedom (base vs joint)

Genesis exposes **full articulation DOFs** (base + joints). GenesisLab’s entity buffers (`joint_pos`, `joint_vel`, …) use **joint space**: base DOFs are stripped, and actuator `dof_indices` refer to that joint-only indexing.

- **Implicit path** (`GenesisOriginalAction`): uses Genesis `joint.dof_start` / full-DOF indices passed to `control_dofs_position`. The articulation joint named `base` is skipped when building the list from the raw entity so base motion is not commanded.
- **Explicit path** (`JointPositionAction`): reads/writes through `ActuatorBase.dof_indices` and `apply_torques`, which add the fixed base offset when calling `control_dofs_force` on the raw entity.

## Two control modes

### 1. Explicit: `JointPositionAction` + named actuator

- Configure an **actuator** on the robot (`RobotCfg.actuators`) and set `JointPositionActionCfg.actuator_name`.
- Each step: build `ArticulationActions` with position targets → `actuator.compute(...)` → torques → `actuator.apply_torques(raw_entity, efforts)`.
- The actuator implementation (e.g. ideal PD, DC motor) owns the **model** gains, limits, and any internal dynamics. You must use an actuator whose `compute` fills `joint_efforts`.

### 2. Implicit: `GenesisOriginalAction` (Genesis built-in PD)

- Targets are sent with **`control_dofs_position`** on the underlying Genesis entity. Genesis applies its **internal PD** for those DOFs.
- No per-step `ActuatorBase.compute` in Python; you only pass desired positions (after scale/offset/clip in this term).
- If the entity has actuators, action size and joint ordering follow actuator definitions; otherwise all non-base joints with DOFs are used.

## Two places PD gains may live

1. **Actuator parameters** — stored on `ActuatorBase` (`stiffness` / `damping`, etc.). These drive **`compute()`** for `JointPositionAction`.
2. **Engine DOF gains** — during `ActuatorManager.setup`, the same actuator stiffness/damping are also written to the Genesis entity via `set_dofs_kp` / `set_dofs_kv` for the matched full-DOF indices. That keeps the simulator’s implicit PD aligned with actuator config when you use APIs such as `control_dofs_position`.

So you conceptually maintain **one actuator config**; it both backs explicit torque computation and syncs engine `kp`/`kv` for implicit-style calls. If you only use `JointPositionAction`, torques come from the actuator object; engine gains are still updated for consistency with Genesis internals.

## Affine map and clipping

- **Both** terms use `target = offset + scale * action` (scalar or per-joint dict where supported).
- `use_default_offset=True` seeds offset from `entity.data.default_joint_pos` for the controlled joints, then adds any configured `offset`.
- **Clip**: `ActionTerm` supports a uniform `(low, high)` tuple or, for `JointPositionAction`, a per-joint dict keyed by patterns (see `ActionTerm._build_clip_bounds`).

## File map

| Module | Role |
|--------|------|
| `joint_actions.py` | `JointPositionAction` / cfgs — explicit actuator pipeline |
| `genesis_original_action.py` | `GenesisOriginalAction` — `control_dofs_position` |
| `_action_common.py` | Shared batching / non-finite checks |

## Configuration quick reference

- **Explicit**: `JointPositionActionCfg(entity_name=..., actuator_name=..., scale=..., offset=..., use_default_offset=..., clip=...)`.
- **Implicit**: `GenesisOriginalActionCfg(entity_name=..., scale=..., offset=..., use_default_offset=..., clip=...)` with tuple `clip` only.

`JointActionCfg.joint_names` is required by the config schema for compatibility with task YAMLs (e.g. `[".*"]`); actuation is still resolved through `actuator_name` and the actuator’s joint list.
