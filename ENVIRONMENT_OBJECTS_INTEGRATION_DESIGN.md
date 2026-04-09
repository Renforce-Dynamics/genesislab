# Environment Objects Integration Design

## Overview

This document describes how the EnvironmentObjectManager integrates into GenesisLab's base environment classes to support interactive scene objects (furniture, props, etc.) that robots can interact with, while maintaining backward compatibility.

## Core Requirements

1. **Joint Space Isolation**: Environment objects' joints must not interfere with robot control space
2. **Load Order**: Objects must be added AFTER robots to preserve robot joint indexing
3. **Backward Compatibility**: Existing environments without environment objects must work unchanged
4. **Manager Pattern**: Follow existing manager patterns for lifecycle and state access
5. **Interactive**: Robots can interact with objects (push chairs, open cabinets, etc.)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ManagerBasedGenesisEnv                   │
│                                                             │
│  __init__():                                                │
│    1. Build scene (LabScene.build())                       │
│       ├─ Create Genesis scene                              │
│       ├─ Add terrain (optional)                            │
│       ├─ Add robots (defines control DOF space)            │
│       ├─ Add environment objects (NEW - separate DOF)      │
│       ├─ Build scene (freeze DOF layout)                   │
│       └─ Initialize robot actuators                        │
│                                                             │
│    2. Load managers (ActionManager, ObservationManager...) │
│       └─ Managers can now access both robots AND objects   │
│                                                             │
│  reset():                                                   │
│    └─ Reset scene controller (robots + objects)            │
│                                                             │
│  step():                                                    │
│    └─ Step physics (robots + objects interact)             │
└─────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. SceneCfg Extension

**File**: `source/genesislab/genesislab/engine/scene/lab_scene_cfg.py`

```python
from genesislab.components.environment_objects import EnvironmentObjectsConfig

@configclass
class SceneCfg:
    # ... existing fields ...
    
    environment_objects: EnvironmentObjectsConfig = None
    """Optional environment objects configuration.
    
    Environment objects are interactive scene elements (furniture, props, etc.)
    that exist independently from robots and terrain. They are loaded AFTER
    robots to avoid joint indexing conflicts.
    
    Example:
        >>> from genesislab.components.environment_objects import (
        ...     EnvironmentObjectsConfig,
        ...     USDObjectCfg,
        ... )
        >>> scene_cfg = SceneCfg(
        ...     environment_objects=EnvironmentObjectsConfig(
        ...         usd_objects=[
        ...             USDObjectCfg(
        ...                 name="furniture",
        ...                 usd_path="scene.usd",
        ...                 load_articulation=True,
        ...             ),
        ...         ],
        ...     ),
        ... )
    """
```

**Backward Compatibility**: Default value is `None`, so existing code works unchanged.

### 2. LabScene Integration

**File**: `source/genesislab/genesislab/engine/scene/lab_scene.py`

Add environment objects storage and access:

```python
class LabScene:
    def __init__(self, cfg: "SceneCfg", device: str = "cuda"):
        # ... existing initialization ...
        self._environment_objects: Dict[str, gs.Entity] = {}
    
    @property
    def environment_objects(self) -> Dict[str, gs.Entity]:
        """Dictionary of environment objects keyed by name."""
        return self._environment_objects
    
    def build(self, env: Any = None) -> None:
        """Build the Genesis scene and entities."""
        # Create Genesis scene
        self._gs_scene = self._scene_builder.create_scene()

        # Add terrain if specified
        if self.cfg.terrain is not None:
            self._terrain = self._scene_builder.add_terrain(self._gs_scene)

        # Add robots (defines control DOF space)
        for entity_name, robot_cfg in self.cfg.robots.items():
            lab_entity = self._scene_builder.add_robot(self._gs_scene, entity_name, robot_cfg, env=env)
            self._entities[entity_name] = lab_entity
        
        # ⭐ ADD ENVIRONMENT OBJECTS (NEW)
        # Objects are added AFTER robots to avoid joint indexing conflicts
        if self.cfg.environment_objects is not None:
            self._environment_objects = self._scene_builder.add_environment_objects(
                self._gs_scene, 
                self.cfg.environment_objects
            )
        
        # Add sensors if specified
        for sensor_name, sensor_cfg in self.cfg.sensors.items():
            self._scene_builder.add_sensor(self, sensor_name, sensor_cfg)
        
        # Build the scene (freezes DOF layout)
        self._scene_builder.build_scene(self._gs_scene)

        # Initialize robot actuators (uses frozen DOF indices)
        for lab_entity in self._entities.values():
            lab_entity.robot_asset.initialize_actuators()
```

**Key Design Decision**: Environment objects are added AFTER robots but BEFORE `scene.build()`. This ensures:
- Robot DOF space is defined first
- Objects use separate DOF space (via `add_stage()`)
- When `scene.build()` is called, both are properly registered
- Robot actuators use correct joint indices

### 3. SceneBuilder Extension

**File**: `source/genesislab/genesislab/engine/scene/scene_builder.py`

Add method to load environment objects:

```python
def add_environment_objects(
    self, 
    scene: gs.Scene, 
    objects_cfg: EnvironmentObjectsConfig
) -> Dict[str, gs.Entity]:
    """Add environment objects to the scene.
    
    This should be called AFTER robots are added but BEFORE scene.build().
    Objects are loaded using add_stage() to maintain separate DOF spaces.
    
    Args:
        scene: Genesis Scene instance.
        objects_cfg: Environment objects configuration.
    
    Returns:
        Dictionary of loaded objects keyed by name.
    """
    from genesislab.components.environment_objects import EnvironmentObjectManager
    
    logger.info("Adding environment objects...")
    
    # Create manager and load objects
    manager = EnvironmentObjectManager(cfg=objects_cfg, scene=scene)
    manager.load_objects()
    
    logger.info(
        f"✅ Added {len(manager.objects)} environment objects: "
        f"{list(manager.objects.keys())}"
    )
    
    # Return objects dictionary for LabScene storage
    return manager.objects
```

### 4. ManagerBasedGenesisEnv - No Changes Required!

**File**: `source/genesislab/genesislab/envs/manager_based_genesis_env.py`

**No modifications needed**. The environment already:
- Builds scene via `LabScene.build()` (which now includes objects)
- Passes `env=self` reference to scene builders
- Managers can access objects via `env.scene.environment_objects`

**Backward Compatibility**: 
- If `cfg.scene.environment_objects` is `None`, no objects are added
- Existing environments work exactly as before

### 5. Accessing Objects from Managers

Observation/reward terms can access environment objects:

```python
# In an observation term
def compute(self, env: ManagerBasedGenesisEnv) -> torch.Tensor:
    # Access objects
    objects = env.scene.environment_objects
    
    # Example: get chair position
    if "chair_01" in objects:
        chair = objects["chair_01"]
        # Access state via Genesis entity API
        chair_pos = chair.get_pos()  # or however Genesis exposes state
    
    return observation
```

## Complete Integration Flow

### 1. Configuration Phase

```python
from genesislab.engine.scene import SceneCfg
from genesislab.components.environment_objects import (
    EnvironmentObjectsConfig,
    USDObjectCfg,
)
from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG

# Define scene with environment objects
scene_cfg = SceneCfg(
    num_envs=4,
    backend="cuda",
    
    # Robots
    robots={
        "humanoid": G1_FULL_ACT_CFG,
    },
    
    # ⭐ Environment objects (NEW)
    environment_objects=EnvironmentObjectsConfig(
        usd_objects=[
            USDObjectCfg(
                name="furniture_scene",
                usd_path="path/to/Scene.usd",
                load_articulation=True,  # Load with joints
            ),
        ],
    ),
)
```

### 2. Initialization Phase

```python
env = ManagerBasedRlEnv(cfg=env_cfg)

# Inside ManagerBasedGenesisEnv.__init__():
# 1. self._scene = LabScene(cfg.scene)
# 2. self._scene.build(env=self)
#    ├─ Add terrain
#    ├─ Add robots (G1 - 23 joints)
#    ├─ Add environment objects (furniture - 254 joints, SEPARATE DOF space)
#    ├─ Build scene (freeze layout)
#    └─ Initialize robot actuators (only accesses G1's 23 joints)
# 3. self._load_managers()
#    └─ Managers can access env.scene.environment_objects
```

### 3. Runtime Phase

```python
# During step()
obs, reward, terminated, truncated, info = env.step(action)

# Physics engine:
# - Applies actions to robot joints (23 DOFs)
# - Simulates environment object joints (254 DOFs) - separate space
# - Handles collisions between robot and objects
# - Objects can move, joints can articulate

# Observation terms can access object state:
# env.scene.environment_objects["furniture_scene"].get_pos()
```

## Lifecycle Management

### Initialization

```
LabScene.build() called
  ↓
SceneBuilder.add_environment_objects()
  ↓
EnvironmentObjectManager created
  ↓
EnvironmentObjectManager.load_objects()
  ├─ Load USD objects via scene.add_stage()
  └─ Load primitive objects via scene.add_entity()
  ↓
Objects stored in LabScene._environment_objects
  ↓
Managers can access via env.scene.environment_objects
```

### Reset

```
env.reset(env_ids)
  ↓
LabScene.controller.reset(env_ids)
  ↓
Genesis resets both robots AND objects
  ↓
Managers reset (can access updated object states)
```

### Step

```
env.step(action)
  ↓
ActionManager applies robot actions
  ↓
Genesis physics step (robots + objects interact)
  ↓
ObservationManager computes obs (can include object state)
  ↓
RewardManager computes rewards (can penalize/reward object interaction)
```

## Joint Space Isolation Strategy

### Problem
When USD objects with joints (254 DOFs) are added before robots (23 DOFs), the global joint space becomes:
- Indices 0-253: Furniture joints
- Indices 254-276: Robot joints (WRONG!)

ActionManager expects robot joints at indices 0-22, causing index errors.

### Solution
**Load Order + add_stage()**:

1. **Add robots first** (via `add_entity()`)
   - Robot joints registered in control DOF space
   - Indices 0-22 for G1 humanoid

2. **Add objects second** (via `add_stage()`)
   - Objects use SEPARATE DOF space
   - Genesis maintains isolation between controllable (robot) and scene (object) DOFs

3. **Build scene**
   - Freezes DOF layout
   - Robot: indices 0-22 (controllable)
   - Objects: separate space (simulated but not in robot control space)

4. **Initialize robot actuators**
   - Safely uses robot joint indices 0-22
   - No conflict with object joints

## Backward Compatibility Checklist

✅ **Existing environments without objects**:
- `SceneCfg.environment_objects` defaults to `None`
- No objects added during scene build
- No performance overhead
- All existing code works unchanged

✅ **Existing managers**:
- No API changes required
- Can optionally access `env.scene.environment_objects` if needed
- Terms without object interaction work exactly as before

✅ **Configuration inheritance**:
- Child configs can add `environment_objects` field
- Parent configs without field remain valid

✅ **Scene building**:
- Scene build order preserved (terrain → robots → build)
- Objects inserted at safe point (after robots, before build)
- No breaking changes to SceneBuilder API

## API Summary

### New Configuration

```python
# In SceneCfg
environment_objects: EnvironmentObjectsConfig = None
```

### New LabScene Properties

```python
# Access environment objects
env.scene.environment_objects  # Dict[str, gs.Entity]

# Check if objects exist
if "chair_01" in env.scene.environment_objects:
    chair = env.scene.environment_objects["chair_01"]
```

### New SceneBuilder Method

```python
# Called internally during LabScene.build()
scene_builder.add_environment_objects(scene, objects_cfg)
```

## Example Usage

### Basic Scene with Objects

```python
from genesislab import ManagerBasedRlEnv
from genesislab.engine.scene import SceneCfg
from genesislab.components.environment_objects import (
    EnvironmentObjectsConfig,
    USDObjectCfg,
    PrimitiveObjectCfg,
)
from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG

# Create environment config
env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
        num_envs=4,
        
        # Add robot
        robots={
            "humanoid": G1_FULL_ACT_CFG,
        },
        
        # Add environment objects
        environment_objects=EnvironmentObjectsConfig(
            # Complete scene with furniture
            usd_objects=[
                USDObjectCfg(
                    name="office_scene",
                    usd_path="path/to/Scene.usd",
                    load_articulation=True,
                ),
            ],
            
            # Additional primitive objects
            primitive_objects=[
                PrimitiveObjectCfg(
                    name="box",
                    shape="box",
                    size=(0.3, 0.3, 0.3),
                    pos=(1.0, 0.0, 0.15),
                ),
            ],
        ),
    ),
    
    # ... observations, actions, rewards, etc ...
)

# Create environment
env = ManagerBasedRlEnv(cfg=env_cfg)

# Objects are automatically loaded and accessible
objects = env.scene.environment_objects
print(f"Loaded objects: {list(objects.keys())}")
```

### Observation Term Using Objects

```python
@configclass
class ObjectDistanceCfg(ObservationTermCfg):
    """Distance to nearest object."""
    
    object_name: str = "chair_01"

def object_distance(env: ManagerBasedGenesisEnv, cfg: ObjectDistanceCfg) -> torch.Tensor:
    """Compute distance from robot to object."""
    # Get robot position
    robot = env.entities["humanoid"]
    robot_pos = robot.data.root_pos_w
    
    # Get object position
    objects = env.scene.environment_objects
    if cfg.object_name not in objects:
        return torch.zeros(env.num_envs, 1, device=env.device)
    
    obj = objects[cfg.object_name]
    obj_pos = obj.get_pos()  # Genesis API for entity position
    
    # Compute distance
    distance = torch.norm(robot_pos - obj_pos, dim=-1, keepdim=True)
    
    return distance
```

## Implementation Checklist

1. ✅ **Create EnvironmentObjectsConfig** (already done)
   - `object_cfg.py` with USDObjectCfg, PrimitiveObjectCfg, EnvironmentObjectsConfig

2. ✅ **Create EnvironmentObjectManager** (already done)
   - `object_manager.py` with load_objects() method

3. ⬜ **Extend SceneCfg**
   - Add `environment_objects: EnvironmentObjectsConfig = None`

4. ⬜ **Extend LabScene**
   - Add `_environment_objects` storage
   - Add `environment_objects` property
   - Modify `build()` to call scene_builder.add_environment_objects()

5. ⬜ **Extend SceneBuilder**
   - Add `add_environment_objects()` method

6. ⬜ **Create Test Script**
   - Test interactive objects with robot
   - Verify no joint indexing conflicts
   - Test backward compatibility

7. ⬜ **Documentation**
   - Update USD implementation guide
   - Add example observation/reward terms
   - Document object state access API

## Success Criteria

✅ **Functional**:
- Complete Scene.usd loads with all furniture (254 joints)
- G1 robot (23 joints) spawns correctly
- No joint indexing errors in ActionManager
- Robot can physically interact with objects (collisions work)
- Objects can be accessed from observation/reward terms

✅ **Performance**:
- No performance degradation for environments without objects
- Multi-environment support works (objects replicated per env)

✅ **Compatibility**:
- All existing test scripts run unchanged
- Existing environments without environment_objects work
- No breaking changes to public APIs

## Open Questions / Future Work

1. **Object State Access**: What's the Genesis API for accessing entity state (position, velocity, joint angles)?
   - Need to document `gs.Entity` API for objects
   - Provide helper methods if needed

2. **Multi-Environment Objects**: How are objects replicated across envs?
   - Need to verify Genesis handles this automatically
   - Document expected behavior

3. **Object Reset**: Can we reset individual object states?
   - Useful for randomization/curriculum
   - May need additional EnvironmentObjectManager methods

4. **Object Interaction Rewards**: Should we provide standard reward terms?
   - Distance to object
   - Object manipulation success
   - etc.

## Conclusion

This design maintains GenesisLab's architecture principles:
- **Manager pattern**: Objects follow same lifecycle as other managers
- **Scene abstraction**: LabScene manages all scene entities
- **Backward compatibility**: Existing code works unchanged
- **Extensibility**: Easy to add object-related terms

The key insight is that objects are **scene entities**, not managers, so they integrate at the scene build level, not the manager level. This allows managers to consume object state just like they consume robot state.
