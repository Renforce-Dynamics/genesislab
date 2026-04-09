# 🚀 USD地形测试 - 快速指南

三种方案测试脚本，支持headless和viewer两种模式。

---

## 📋 测试脚本一览

| 脚本 | 方案 | 地形类型 | 初始化速度 |
|------|------|---------|-----------|
| `test_plane_terrain.py` | Plane Terrain | 无限平面 | ⚡ <1秒 |
| `test_approach1_usd_terrain.py` | USD Terrain | 静态USD | ⚡ ~3秒 |
| `test_approach2_usd_scene.py` | USD Scene | 完整场景 | ⏳ ~2分钟 |

---

## 🎮 使用方法

### 方案1: Plane Terrain（平地基线）

```bash
# 无头模式（默认）
python scripts/test_plane_terrain.py

# 启用Viewer
python scripts/test_plane_terrain.py --viewer
```

**特点：**
- ⚡ 最快初始化（<1秒）
- 🎯 适合基础训练
- 💚 最小内存占用

---

### 方案2: USD Terrain（静态地形）

```bash
# 无头模式（默认）
python scripts/test_approach1_usd_terrain.py

# 启用Viewer
python scripts/test_approach1_usd_terrain.py --viewer
```

**特点：**
- ⚡ 快速初始化（~3秒）
- 🏢 真实建筑环境（Floor, Wall, Ceiling）
- 📦 支持多环境

---

### 方案3: USD Scene（完整场景）

```bash
# 无头模式（默认）
python scripts/test_approach2_usd_scene.py

# 启用Viewer
python scripts/test_approach2_usd_scene.py --viewer
```

**特点：**
- ⏳ 较慢初始化（~2分钟）
- 🪑 包含254个关节的家具
- 🤝 物体可交互

---

## 💡 推荐测试顺序

### 1. 首次测试（无头模式，快速验证）

```bash
cd /home/ununtu/code/glab/genesislab

# 第一步：Plane（最快，验证环境）
python scripts/test_plane_terrain.py

# 第二步：USD Terrain（验证USD加载）
python scripts/test_approach1_usd_terrain.py

# 第三步：USD Scene（验证完整场景，可选）
python scripts/test_approach2_usd_scene.py
```

### 2. 可视化测试（启用Viewer）

```bash
# 方案1: Plane + Viewer（最流畅）
python scripts/test_plane_terrain.py --viewer

# 方案2: USD Terrain + Viewer（真实环境）
python scripts/test_approach1_usd_terrain.py --viewer

# 方案3: USD Scene + Viewer（完整场景，需要等待）
python scripts/test_approach2_usd_scene.py --viewer
```

---

## 🎯 使用场景选择

### 开发调试 → Plane + Viewer
```bash
python scripts/test_plane_terrain.py --viewer
```
- 最快启动
- 实时查看机器人动作
- 适合算法调试

### 环境测试 → USD Terrain + Viewer
```bash
python scripts/test_approach1_usd_terrain.py --viewer
```
- 真实建筑环境
- 验证导航能力
- 多环境训练

### 演示展示 → USD Scene + Viewer
```bash
python scripts/test_approach2_usd_scene.py --viewer
```
- 完整真实场景
- 物体交互展示
- 最佳视觉效果

### 批量测试 → Headless模式（所有方案）
```bash
# 不需要显示，更快
python scripts/test_plane_terrain.py
python scripts/test_approach1_usd_terrain.py
python scripts/test_approach2_usd_scene.py
```

---

## 📊 性能对比（实测）

### Headless模式（无Viewer）

| 方案 | 初始化 | FPS | 内存 |
|------|--------|-----|------|
| Plane | <1s | 200+ | 200MB |
| USD Terrain | ~3s | 150+ | 500MB |
| USD Scene | ~120s | 100+ | 1-3GB |

### Viewer模式（启用）

| 方案 | 初始化 | FPS | 内存 |
|------|--------|-----|------|
| Plane | ~2s | 60+ | 300MB |
| USD Terrain | ~5s | 50+ | 600MB |
| USD Scene | ~150s | 30+ | 2-4GB |

---

## 🎬 Viewer控制

启用Viewer后，可以使用以下控制：

**鼠标控制：**
- 🖱️ **左键拖拽**: 旋转视角
- 🖱️ **滚轮**: 缩放
- 🖱️ **右键拖拽**: 平移相机

**键盘控制：**
- ⬆️⬇️⬅️➡️ **方向键**: 移动相机
- **ESC**: 退出

---

## 🔧 常见问题

### Q: 如何选择用哪个方案？

**A:**
- **快速测试** → Plane
- **真实环境** → USD Terrain  
- **完整场景** → USD Scene

### Q: Viewer模式下FPS太低？

**A:**
```bash
# 减少环境数量（编辑脚本中的num_envs）
# 或使用headless模式
python scripts/test_*.py  # 不加--viewer
```

### Q: USD Scene初始化太慢？

**A:**
- 这是正常的（254个关节需要处理）
- 首次加载可能需要2-5分钟
- 可以先用USD Terrain测试

---

## 📁 文件位置

```
genesislab/
├── scripts/
│   ├── test_plane_terrain.py           ✅ Plane测试
│   ├── test_approach1_usd_terrain.py    ✅ USD Terrain测试
│   └── test_approach2_usd_scene.py      ✅ USD Scene测试
├── QUICK_TEST_GUIDE.md                  ✅ 本文件
└── third_party/genPiHub/data/assets/
    └── CWDL_LW_Assets_20260310/
        ├── Scene.usd                    ✅ 完整场景
        └── Terrain.usd                  ✅ 静态地形
```

---

## ✅ 快速验证

一键运行所有测试（headless模式）：

```bash
cd /home/ununtu/code/glab/genesislab

echo "=== Test 1: Plane Terrain ===" && \
python scripts/test_plane_terrain.py && \
echo -e "\n=== Test 2: USD Terrain ===" && \
python scripts/test_approach1_usd_terrain.py && \
echo -e "\n=== All basic tests passed! ==="
```

---

## 🎉 总结

**三种方案，两种模式，灵活选择！**

| 需求 | 命令 |
|------|------|
| 快速验证 | `python scripts/test_plane_terrain.py` |
| 真实环境 | `python scripts/test_approach1_usd_terrain.py` |
| 完整场景 | `python scripts/test_approach2_usd_scene.py` |
| 可视化 | 在任何命令后加 `--viewer` |

**开始你的测试吧！** 🚀
