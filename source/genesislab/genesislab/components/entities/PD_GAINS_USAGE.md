# PD Gains Usage in GenesisLab

## Overview

This document explains how PD (Proportional-Derivative) gains work in GenesisLab and where they are applied.

## Two Separate PD Systems

GenesisLab has **two separate PD control systems** that serve different purposes:

### 1. Engine-Level PD Control (RobotCfg)

**Location**: `genesislab.components.entities.robot_cfg.RobotCfg`

**Fields**:
- `pd_gains: dict[str, tuple[float, float]]` - Per-joint PD gains as `{joint_name_pattern: (kp, kd)}`
- `default_pd_kp: float` - Uniform proportional gain for all DOFs
- `default_pd_kd: float` - Uniform derivative gain for all DOFs

**Where it's used**: 
- `GenesisBinding._apply_robot_pd_gains()` reads these values
- Applied via `entity.set_dofs_kp()` and `entity.set_dofs_kv()` directly to Genesis engine
- This configures the **engine-level PD controller** for position control

**Priority**:
1. If `pd_gains` (per-joint dict) is specified, those values are used (supports regex patterns)
2. Otherwise, if `default_pd_kp` and `default_pd_kd` are set, uniform gains are applied to all DOFs

**Example**:
```python
robot_cfg = RobotCfg(
    # ... other config ...
    pd_gains={
        ".*_hip_pitch_joint": (180.0, 18.0),  # Regex pattern matching
        ".*_knee_joint": (150.0, 15.0),
    },
    # OR use uniform gains:
    # default_pd_kp=100.0,
    # default_pd_kd=10.0,
)
```

### 2. Actuator-Level PD Control (ActuatorBaseCfg)

**Location**: `genesislab.components.actuators.actuator_base_cfg.ActuatorBaseCfg`

**Fields**:
- `stiffness: dict[str, float] | float` - PD stiffness (kp) for actuator model
- `damping: dict[str, float] | float` - PD damping (kd) for actuator model

**Where it's used**:
- Actuator models (e.g., `IdealPDActuator`, `ImplicitActuator`) use these for torque computation
- For **implicit actuators**: These values are set to the engine (similar to RobotCfg PD gains)
- For **explicit actuators**: These values are used internally by the actuator model to compute torques

**Note**: 
- Currently, GenesisLab's `RobotCfg` does **not** have an `actuators` field
- Actuator configurations are separate and would be used in a higher-level `ArticulationCfg` (similar to IsaacLab)
- The actuator system is available but not automatically connected to `RobotCfg`

## Current Implementation Status

### ✅ Implemented
- `default_pd_kp` and `default_pd_kd` in `RobotCfg` → Applied to Genesis engine (legacy system)
- `pd_gains` (per-joint dict) in `RobotCfg` → Applied to Genesis engine (with regex pattern matching, legacy system)
- **`actuators` in `RobotCfg` → IsaacLab-style actuator system (NEW!)**
  - Supports both implicit and explicit actuator models
  - For implicit actuators: Sets stiffness/damping to Genesis engine (similar to legacy PD gains)
  - For explicit actuators: Sets engine kp/kv to 0, actuator computes torques explicitly
  - Takes precedence over legacy PD gains when configured

### ✅ Genesis Engine's Built-in PD Controller

**Yes, Genesis engine internally implements PD control!**

When you call `entity.control_dofs_position(targets)`, the Genesis engine automatically computes control forces using the PD gains you've set:

```python
force = kp * (target_pos - current_pos) + kv * (target_vel - current_vel)
```

This is done **inside the physics engine** during the simulation step, which means:
- The PD control is **implicit** (handled by the solver)
- It uses **continuous-time integration** for better accuracy
- The control force is automatically applied each step

This is different from **explicit PD control** (like `IdealPDActuator`), where you manually compute torques in Python and then apply them.

### ✅ Actuator System (IsaacLab-Style)

**Now fully implemented!**

You can now use IsaacLab-style actuator configurations in `RobotCfg`:

```python
from genesislab.components.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg

robot_cfg = RobotCfg(
    # ... other config ...
    actuators={
        "default": IdealPDActuatorCfg(
            joint_names_expr=[".*"],  # All joints
            stiffness=100.0,
            damping=10.0,
        )
    }
)
```

**How it works:**
1. **Implicit Actuators** (e.g., `ImplicitActuatorCfg`):
   - Stiffness/damping are set directly to Genesis engine (via `set_dofs_kp()`/`set_dofs_kv()`)
   - Engine handles PD control internally (same as legacy PD gains)
   - More accurate for large time steps (continuous-time integration)

2. **Explicit Actuators** (e.g., `IdealPDActuatorCfg`, `DelayedPDActuatorCfg`, `UnitreeActuatorCfg`):
   - Engine kp/kv are set to 0 (no engine-level PD control)
   - Actuator model computes torques explicitly in Python
   - Supports complex dynamics (delays, torque-speed curves, etc.)
   - Torques are applied via `control_dofs_force()` (to be implemented in action manager)

**Priority:**
- If `actuators` is configured → Actuator system is used (legacy PD gains are skipped)
- If `actuators` is `None` → Legacy PD gains system is used (`pd_gains` or `default_pd_kp/kd`)

## How PD Gains Are Applied

1. **During Scene Building** (`GenesisBinding.build()`):
   - After scene is built, `_process_actuators_cfg()` is called first (if actuators are configured)
   - Then `_apply_robot_pd_gains()` is called (only if actuators are NOT configured)
   - PD gains are set via `entity.set_dofs_kp()` and `entity.set_dofs_kv()`

2. **Actuator System** (if `RobotCfg.actuators` is configured):
   - **Implicit actuators**: Stiffness/damping from actuator config → Set to engine kp/kv
   - **Explicit actuators**: Engine kp/kv set to 0, actuator computes torques in Python

3. **Legacy PD Gains System** (if `RobotCfg.actuators` is `None`):
   - Processes `RobotCfg.pd_gains` or `RobotCfg.default_pd_kp/kd`
   - Sets engine kp/kv directly

4. **Engine-Level Control**:
   - When you call `entity.control_dofs_position(targets)`, Genesis engine uses the PD gains
   - The engine computes: `torque = kp * (target - current_pos) + kd * (target_vel - current_vel)`
   - For explicit actuators, you need to call `entity.control_dofs_force()` with computed torques

## Summary

- **Genesis Engine Built-in PD**: Yes! Genesis has a built-in PD controller that automatically computes control forces when you use `control_dofs_position()` or `control_dofs_velocity()`
- **RobotCfg PD gains (legacy)** → Directly set to Genesis engine via `set_dofs_kp()`/`set_dofs_kv()` → Used by engine's built-in PD controller
- **RobotCfg.actuators (NEW!)** → IsaacLab-style actuator system:
  - **Implicit actuators**: Stiffness/damping → Set to engine kp/kv → Engine handles PD control
  - **Explicit actuators**: Engine kp/kv = 0 → Actuator computes torques → Applied via `control_dofs_force()`
- **Priority**: Actuators take precedence over legacy PD gains when configured

## Key Insight

**Genesis engine = Implicit PD control (built-in)**
- When you set `kp`/`kv` and call `control_dofs_position()`, the engine automatically computes and applies PD forces
- This is similar to IsaacLab's "implicit actuators" - the PD control is handled by the physics solver

**Actuator models = Explicit PD control (IsaacLab-style, now supported!)**
- For more complex actuator dynamics (delays, torque-speed curves, etc.), you can use explicit actuator models
- These compute torques in Python and then apply them as force commands via `control_dofs_force()`
- **Now fully integrated**: Configure via `RobotCfg.actuators` and the system handles everything automatically

## Usage Example

```python
from genesislab.components.actuators import IdealPDActuatorCfg

# Use explicit PD actuator (computes torques in Python)
robot_cfg = RobotCfg(
    morph_type="URDF",
    morph_path="path/to/robot.urdf",
    actuators={
        "default": IdealPDActuatorCfg(
            joint_names_expr=[".*"],  # All joints
            stiffness=100.0,
            damping=10.0,
        )
    }
)

# Or use implicit actuator (engine handles PD control)
from genesislab.components.actuators import ImplicitActuatorCfg

robot_cfg = RobotCfg(
    morph_type="URDF",
    morph_path="path/to/robot.urdf",
    actuators={
        "default": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=100.0,
            damping=10.0,
        )
    }
)
```
