# 🎯 USD Integration in GenesisLab - Complete Guide

GenesisLab支持两种方式使用USD场景文件。

---

## 📋 两种实现方案对比

| 特性 | 方案1: USD Terrain | 方案2: USD Scene |
|------|-------------------|------------------|
| **用途** | 纯静态地形 | 完整场景（含家具） |
| **USD要求** | 必须无关节 | 可以有关节物体 |
| **多环境支持** | ✅ 网格布局 | ⚠️ 通常单环境 |
| **地形系统** | ✅ 集成 | ❌ 独立 |
| **物体交互** | ❌ 静态 | ✅ 可交互 |
| **使用场景** | 建筑环境行走 | 复杂场景操作 |

---

## 方案1: USD作为Terrain

### 适用场景
- ✅ 纯静态几何体（地面、墙壁、天花板）
- ✅ 需要多环境并行测试
- ✅ 建筑导航任务
- ❌ 不适合有关节的物体（家具、门等）

### 使用方法

```python
from genesislab import LabScene, SceneCfg
from genesislab.components.terrains import TerrainCfg

cfg = SceneCfg(
    num_envs=4,  # 多环境
    terrain=TerrainCfg(
        terrain_type="usd",
        usd_path="path/to/static_terrain.usd",  # 纯静态USD
        env_spacing=10.0,  # 环境间距
    ),
)

scene = LabScene(cfg)
scene.build()
```

### USD文件要求

**✅ 可以包含：**
- 静态mesh（Floor, Wall, Ceiling等）
- Visuals和Collisions
- 材质和纹理

**❌ 不能包含：**
- PhysicsJoint（关节）
- 可动部件（doors, drawers等）
- Articulated objects

### 创建静态USD的方法

如果你的Scene.usd包含关节物体，使用提供的工具提取静态部分：

```bash
python third_party/genPiHub/scripts/utils/extract_terrain_from_scene.py
```

这会创建一个`Terrain.usd`，只包含Floor, Wall, Ceiling等静态元素。

---

## 方案2: USD作为Scene Entities

### 适用场景
- ✅ 完整场景（包含家具、道具）
- ✅ 可动物体（椅子、柜子等）
- ✅ 机器人与场景交互
- ✅ 单环境详细仿真

### 使用方法

```python
from genesislab import LabScene, SceneCfg
from genesislab.components.terrains import TerrainCfg

cfg = SceneCfg(
    num_envs=1,  # 通常单环境
    # 加载完整USD场景（含家具）
    usd_scene_path="path/to/complete_scene.usd",
    # 可选：添加简单地形用于机器人生成
    terrain=TerrainCfg(terrain_type="plane"),
)

scene = LabScene(cfg)
scene.build()
```

### USD文件要求

**✅ 可以包含：**
- 所有方案1的内容
- ✅ PhysicsJoint（关节）
- ✅ 可动物体
- ✅ Articulated objects
- ✅ 复杂场景结构

**特点：**
- 使用`scene.add_stage()`加载，支持混合实体
- 物体可以与机器人交互
- 适合操作任务、场景交互

---

## 🔧 技术实现细节

### 方案1实现（scene_builder.py）

```python
def _add_usd_terrain(self, scene: gs.Scene, terrain_cfg) -> TerrainRuntime:
    """Add USD terrain using add_stage for mixed entity support."""
    morph = gs.morphs.USD(file=terrain_cfg.usd_path)
    scene.add_stage(morph=morph)  # 使用add_stage处理复杂USD
    
    # 返回grid-based环境origins
    return TerrainRuntime(
        terrain_generator=None,
        terrain_origins=None,
        num_envs=scene_cfg.num_envs,
        env_spacing=env_spacing,
        device=self._scene.device,
    )
```

### 方案2实现（scene_builder.py）

```python
def add_usd_scene(self, scene: gs.Scene) -> None:
    """Load USD scene as background environment entities."""
    morph = gs.morphs.USD(file=self._scene.cfg.usd_scene_path)
    scene.add_stage(morph=morph)  # 支持articulated objects
```

**调用顺序（lab_scene.py）：**
```python
1. create_scene()       # 创建Genesis场景
2. add_usd_scene()      # 加载USD场景实体（如果指定）
3. add_terrain()        # 添加地形（如果指定）
4. add_robot()          # 添加机器人
```

---

## 📁 文件结构

```
genesislab/
├── source/genesislab/genesislab/
│   ├── engine/scene/
│   │   ├── scene_builder.py          # ✅ 实现两种方案
│   │   └── lab_scene_cfg.py          # ✅ 添加usd_scene_path
│   └── components/terrains/
│       └── terrain_cfg.py             # ✅ 支持terrain_type="usd"
├── scripts/
│   ├── demo_usd_approaches.py         # 📖 演示两种方案
│   └── test_usd_terrain_*.py          # 🧪 测试脚本
└── third_party/genPiHub/
    ├── scripts/amo/
    │   ├── play_amo_usd_terrain.py    # 🎬 AMO+USD示例
    │   └── utils/
    │       └── extract_terrain_from_scene.py  # 🔧 提取静态地形
    └── data/assets/CWDL_LW_Assets_20260310/
        ├── Scene.usd                   # 完整场景（含家具）
        └── Terrain.usd                 # 静态地形（生成的）
```

---

## 🚀 快速开始

### 测试方案1（USD Terrain）

```bash
cd /home/ununtu/code/glab/genesislab

# 1. 提取静态地形（如果需要）
python third_party/genPiHub/scripts/utils/extract_terrain_from_scene.py

# 2. 运行演示
python scripts/demo_usd_approaches.py --approach terrain
```

### 测试方案2（USD Scene）

```bash
python scripts/demo_usd_approaches.py --approach scene
```

### AMO策略 + USD地形

```bash
# 使用静态地形（方案1）
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
    --viewer \
    --usd-path third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Terrain.usd
```

---

## 💡 最佳实践

### 何时使用方案1（USD Terrain）

1. **建筑导航任务**
   ```python
   terrain=TerrainCfg(
       terrain_type="usd",
       usd_path="warehouse_floor.usd",  # 仓库地面
   )
   ```

2. **多环境训练**
   ```python
   num_envs=16  # 16个并行环境
   terrain=TerrainCfg(terrain_type="usd", env_spacing=15.0)
   ```

3. **性能优化**
   - 静态场景加载更快
   - 多环境内存效率高

### 何时使用方案2（USD Scene）

1. **操作任务**
   ```python
   usd_scene_path="kitchen_with_cabinets.usd"  # 厨房+柜子
   # 机器人可以打开柜门
   ```

2. **场景交互**
   ```python
   usd_scene_path="office_with_furniture.usd"  # 办公室+椅子
   # 机器人可以移动椅子
   ```

3. **单环境高保真仿真**
   - 完整物理交互
   - 复杂动态场景

### 组合使用

可以同时使用两种方案：

```python
cfg = SceneCfg(
    usd_scene_path="furniture.usd",      # 家具（可动）
    terrain=TerrainCfg(
        terrain_type="usd",
        usd_path="building_structure.usd"  # 建筑结构（静态）
    ),
)
```

---

## 🐛 常见问题

### Q1: "Mixed entity detected" 错误

**问题**: USD包含both articulated objects和independent rigid bodies

**解决方案**:
- **方案1**: 提取静态部分到新USD
  ```bash
  python scripts/utils/extract_terrain_from_scene.py
  ```
- **方案2**: 使用`usd_scene_path`代替terrain
  ```python
  usd_scene_path="Scene.usd"  # 使用scene方式
  ```

### Q2: 关节索引错误

**问题**: 加载USD后机器人关节索引错乱

**原因**: USD中的关节被计入scene的关节总数

**解决方案**: 使用方案1（提取静态USD）或方案2（usd_scene_path）

### Q3: USD文件找不到

**检查**:
```bash
ls third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd
```

**使用绝对路径**:
```python
usd_path="/absolute/path/to/Scene.usd"
```

---

## ✅ 验证清单

- [x] 方案1: USD Terrain实现完成
- [x] 方案2: USD Scene实现完成
- [x] scene.add_stage()支持mixed entities
- [x] 静态USD提取工具
- [x] AMO + USD测试脚本
- [x] 演示脚本（两种方案）
- [x] 完整文档

---

## 📚 相关文档

- [AMO_USD_TERRAIN_GUIDE.md](AMO_USD_TERRAIN_GUIDE.md) - AMO策略+USD地形指南
- [QUICK_REFERENCE_AMO_USD.md](QUICK_REFERENCE_AMO_USD.md) - 快速参考
- [scripts/README_USD_TERRAIN.md](scripts/README_USD_TERRAIN.md) - 测试脚本文档

---

## 🎉 总结

GenesisLab现在支持两种灵活的USD集成方式：

1. **USD Terrain**: 静态地形，多环境，高性能
2. **USD Scene**: 完整场景，可交互，高保真

选择合适的方案，根据你的任务需求！

**实施日期**: 2026-04-09  
**状态**: ✅ Production Ready
