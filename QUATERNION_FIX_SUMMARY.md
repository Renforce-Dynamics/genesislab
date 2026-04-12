# Quaternion Format Fix - Solution A (Unified WXYZ)

## 📋 Summary

**Adopted Solution**: Unified WXYZ format throughout the entire codebase

All quaternions now use **[w, x, y, z]** (wxyz/scalar-first) format consistently:
- ✅ Configuration files
- ✅ Math utilities  
- ✅ Genesis API calls
- ✅ Reset functions
- ✅ Robot configs

**No format conversion needed** - everything is wxyz!

---

## 🔧 Changed Files

### Core Framework

1. **articulation_cfg.py** (Config definition)
   - Default changed: `[0,0,0,1]` → `[1,0,0,0]`
   - Docstring updated: Now states "wxyz format"
   - Identity is `[1, 0, 0, 0]`

2. **lab_entity_data.py** (Default quaternion)
   - Updated `default_root_quat_w` property
   - Now uses `[1,0,0,0]` as wxyz identity
   - No conversion code needed

3. **articulation.py** (Build & Reset)
   - `build_into_scene()`: Uses wxyz default `[1,0,0,0]`
   - `reset()`: Uses wxyz default `[1,0,0,0]`
   - Directly passes to Genesis API

4. **reset.py** (Rotation offsets)
   - **Fixed bug**: Rotation offsets now properly applied
   - Uses `quat_from_euler_xyz()` and `quat_mul()`
   - All operations in wxyz format

5. **scene_controller.py** (API docs)
   - Updated docstring to clarify wxyz format requirement

### Robot Configurations

All robot configs updated to wxyz format:

- ✅ `unitree/go2.py` - `quat=[1.0, 0.0, 0.0, 0.0]`
- ✅ `unitree/b2.py` - `quat=[1.0, 0.0, 0.0, 0.0]`
- ✅ `unitree/h1.py` - `quat=[1.0, 0.0, 0.0, 0.0]`
- ✅ `booster/k1.py` - `quat=[1.0, 0.0, 0.0, 0.0]`
- ✅ `booster/t1.py` - `quat=[1.0, 0.0, 0.0, 0.0]`
- ✅ `g1/official.py` - `quat=[1.0, 0.0, 0.0, 0.0]`
- ✅ `g1/beyondmimic.py` - `quat=[1.0, 0.0, 0.0, 0.0]`
- ✅ `smpl/smpl.py` - `quat=[1.0, 0.0, 0.0, 0.0]`
- ✅ `smpl/smplx.py` - `quat=[1.0, 0.0, 0.0, 0.0]`

---

## 🐛 Bugs Fixed

### 1. Format Inconsistency (CRITICAL)
**Before**: Config used xyzw `[0,0,0,1]`, Genesis expected wxyz `[w,x,y,z]`  
**After**: Everything uses wxyz `[1,0,0,0]`

### 2. Rotation Offsets Not Applied (BUG)
**Before**: `rot_offsets` sampled but never used in `reset_root_state_uniform()`  
**After**: Properly converted to quaternion and multiplied

**Code Added**:
```python
# Apply rotation offsets to quaternion
if rot_offsets.abs().sum() > 0:
    roll, pitch, yaw = rot_offsets.unbind(-1)
    quat_offset = quat_from_euler_xyz(roll, pitch, yaw)  # wxyz
    quat_w = quat_mul(quat_w, quat_offset)  # Apply offset
```

---

## 🧪 Testing

### Test Suite Created
`scripts/test/test_quaternion_consistency.py`

**All Tests Passed** ✅
- Config default is wxyz `[1,0,0,0]`
- Math utilities work with wxyz
- Quaternion application correct
- Quaternion inverse correct
- Reset offset application works
- No xyzw remnants

### Manual Verification
```bash
# Run consistency tests
python scripts/test/test_quaternion_consistency.py

# Check robot configs
grep -r "quat=\[1\.0, 0\.0, 0\.0, 0\.0\]" source/genesis_assets/
```

---

## 📖 Documentation

### Quaternion Format Guide

**Identity Quaternion** (no rotation):
```python
quat = [1.0, 0.0, 0.0, 0.0]  # wxyz: w=1, xyz=0
```

**90° Rotation about Z**:
```python
# From Euler angles
roll, pitch, yaw = 0, 0, 1.5708  # 90° in radians
quat = quat_from_euler_xyz(roll, pitch, yaw)
# Result: [0.707, 0, 0, 0.707]  # wxyz
```

**Applying Rotation**:
```python
# Multiply quaternions (order matters!)
quat_result = quat_mul(quat_base, quat_offset)
```

**Rotating a Vector**:
```python
vec_rotated = quat_apply(quat, vec)
```

### Config Examples

```python
from genesislab.engine.assets.articulation import InitialPoseCfg

# Identity (no rotation)
pose = InitialPoseCfg(
    pos=[0.0, 0.0, 0.5],
    quat=[1.0, 0.0, 0.0, 0.0]  # wxyz identity
)

# 30° yaw (rotation about Z)
pose = InitialPoseCfg(
    pos=[0.0, 0.0, 0.5],
    quat=[0.9659, 0.0, 0.0, 0.2588]  # wxyz: 30° about Z
)
```

---

## ⚠️ Migration Guide

### For Existing Users

**If you used default configs** (identity quaternion):
- ✅ **No changes needed** - defaults automatically updated

**If you manually specified quaternions in xyzw**:
- ❌ Old: `quat=[x, y, z, w]` e.g. `[0, 0, 0, 1]`
- ✅ New: `quat=[w, x, y, z]` e.g. `[1, 0, 0, 0]`

**Conversion**:
```python
# If you have old xyzw format
quat_xyzw = [0.0, 0.0, 0.707, 0.707]  # 90° Z in xyzw

# Convert to wxyz
quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
# Result: [0.707, 0.0, 0.0, 0.707]  # 90° Z in wxyz
```

### For Developers

**When adding new robots**:
```python
initial_pose = InitialPoseCfg(
    pos=[0.0, 0.0, height],
    quat=[1.0, 0.0, 0.0, 0.0]  # Always use wxyz identity
)
```

**When using quaternions in code**:
- All quaternions are wxyz `[w, x, y, z]`
- Use `quat_from_euler_xyz()` to create from Euler angles
- Use `quat_mul()` to combine rotations
- Use `quat_apply()` to rotate vectors

---

## 🎯 Impact

### Benefits
- ✅ **Consistent format** throughout codebase
- ✅ **No conversion overhead** - direct API calls
- ✅ **Matches Genesis API** - wxyz is standard
- ✅ **Matches math utilities** - all use wxyz
- ✅ **Bug fixes** - rotation offsets now work

### Potential Issues (None Expected)
- Most users used default configs (unaffected)
- Custom quaternions rare (easy to update)
- Identity quaternion value changed but semantic same

---

## 📚 References

### Math Utilities (`utils/math/rotation.py`)
All functions use **wxyz format**:
- `quat_from_euler_xyz(roll, pitch, yaw)` → wxyz quaternion
- `quat_mul(q1, q2)` → wxyz quaternion multiplication
- `quat_apply(quat, vec)` → rotated vector
- `quat_inv(quat)` → inverse quaternion
- `quat_apply_inverse(quat, vec)` → inverse rotation

### Genesis API
- `entity.get_quat()` → returns wxyz
- `entity.set_quat(quat)` → expects wxyz
- Morphs (`URDF`, `MJCF`, `USD`) → expect wxyz

### Related Files
- Config: `engine/assets/articulation/articulation_cfg.py`
- Reset: `envs/mdp/events/reset.py`
- Math: `utils/math/rotation.py`
- Tests: `scripts/test/test_quaternion_consistency.py`

---

## ✅ Verification Checklist

- [x] Config default is wxyz `[1,0,0,0]`
- [x] All robot configs updated to wxyz
- [x] No format conversion code
- [x] Math utilities use wxyz
- [x] Reset functions apply rotation offsets
- [x] Documentation updated
- [x] Tests passing
- [x] No xyzw remnants

---

**Date**: 2026-04-11  
**Status**: ✅ Complete  
**Solution**: Unified WXYZ format (Solution A)
