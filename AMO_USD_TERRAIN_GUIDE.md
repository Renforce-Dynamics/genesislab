# 🏃 AMO Policy on USD Terrain - Complete Guide

将AMO人形机器人策略部署到Scene.usd自定义地形上的完整指南。

---

## 🎯 这是什么？

这个功能让你能够：
- ✅ 在真实的3D场景（Scene.usd）中运行AMO策略
- ✅ 看到人形机器人在建筑环境中行走
- ✅ 实时控制机器人的移动方向和速度
- ✅ 同时运行多个机器人环境

**场景文件**: `third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd`

---

## 🚀 快速开始（3步）

### 步骤1: 验证环境

```bash
cd /home/ununtu/code/glab/genesislab

# 检查USD文件是否存在
ls -lh third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd

# 检查AMO模型文件
ls -lh data/AMO/
```

确保你有这些文件：
- ✅ `Scene.usd` - USD地形文件
- ✅ `data/AMO/amo_jit.pt` - AMO策略模型
- ✅ `data/AMO/adapter_jit.pt` - 适配器模型
- ✅ `data/AMO/adapter_norm_stats.pt` - 归一化统计

---

### 步骤2: 运行基础测试

```bash
# 快速测试（不需要viewer，50步）
./third_party/genPiHub/scripts/amo/test_amo_usd_quick.sh
```

**期望输出：**
```
✅ USD file found
✅ Headless test: PASSED
🎉 AMO on USD terrain integration is working!
```

---

### 步骤3: 启动可视化

```bash
# 带viewer运行（推荐！）
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py --viewer
```

**你会看到：**
- 🎬 3D可视化窗口
- 🤖 人形机器人站在Scene.usd地形上
- 🏃 机器人开始向前行走（vx=0.3m/s）

---

## 🎮 交互式控制

想要手动控制机器人？启用交互模式：

```bash
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py --viewer --interactive
```

### 键盘控制：

| 按键 | 功能 | 说明 |
|------|------|------|
| `W` | 前进 | 增加前向速度 |
| `S` | 后退 | 减少前向速度 |
| `A` | 左转 | 增加左转角速度 |
| `D` | 右转 | 增加右转角速度 |
| `E` | 左平移 | 增加左侧向速度 |
| `C` | 右平移 | 增加右侧向速度 |
| `Z` | 升高 | 增加身体高度 |
| `X` | 降低 | 减少身体高度 |
| `Q` | 退出 | 停止程序 |

### Viewer控制：

- 🖱️ **鼠标拖拽**: 旋转视角
- 🖱️ **鼠标滚轮**: 缩放
- ⌨️ **方向键**: 平移相机

---

## 📝 使用场景

### 场景1: 单机器人探索

```bash
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
    --viewer \
    --interactive \
    --num-envs 1
```

**用途**: 详细观察单个机器人的行为

---

### 场景2: 固定速度前进

```bash
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
    --viewer \
    --vx 0.5 \
    --num-envs 1
```

**用途**: 测试特定速度下的性能

---

### 场景3: 多机器人并行

```bash
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
    --viewer \
    --num-envs 4 \
    --env-spacing 12.0 \
    --vx 0.3
```

**用途**: 同时测试多个机器人实例

**环境布局：**
```
Env 0: [  0.00,   0.00]
Env 1: [ 12.00,   0.00]
Env 2: [  0.00,  12.00]
Env 3: [ 12.00,  12.00]
```

---

### 场景4: 自定义USD地形

```bash
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
    --viewer \
    --usd-path /path/to/your/custom_terrain.usd \
    --env-spacing 8.0
```

**用途**: 在自己设计的地形上测试

---

### 场景5: 无头模式测试

```bash
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
    --headless \
    --max-steps 1000 \
    --print-every 100
```

**用途**: 服务器上批量测试，不需要图形界面

---

## ⚙️ 完整参数说明

### 地形参数

```bash
--usd-path <PATH>          # USD地形文件路径
                           # 默认: third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd

--env-spacing <METERS>     # 环境间距（米）
                           # 默认: 10.0
                           # 建议: 根据地形大小调整
```

### 环境参数

```bash
--num-envs <N>             # 环境数量
                           # 默认: 1
                           # 建议: 1-16（取决于GPU性能）

--device <cuda|cpu>        # 计算设备
                           # 默认: 自动检测（优先cuda）

--viewer                   # 启用3D可视化
--headless                 # 禁用可视化（服务器模式）
```

### 控制参数

```bash
--interactive              # 启用键盘交互控制

--vx <M/S>                 # 前向速度（米/秒）
                           # 默认: 0.3
                           # 范围: -1.0 ~ 1.0

--vy <M/S>                 # 侧向速度（米/秒）
                           # 默认: 0.0
                           # 范围: -0.6 ~ 0.6

--yaw-rate <RAD/S>         # 转向速度（弧度/秒）
                           # 默认: 0.0
                           # 范围: -1.0 ~ 1.0

--height <M>               # 高度调整（米）
                           # 默认: 0.0
```

### 仿真参数

```bash
--max-steps <N>            # 最大仿真步数
                           # 默认: 100000

--print-every <N>          # 每N步打印一次状态
                           # 默认: 100

--action-scale <SCALE>     # 动作缩放系数
                           # 默认: 0.25
                           # 影响动作幅度
```

### 策略参数

```bash
--model-dir <PATH>         # AMO模型目录
                           # 默认: data/AMO
```

---

## 📊 性能优化

### 单环境（最佳视觉效果）

```bash
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
    --viewer \
    --num-envs 1 \
    --device cuda
```

- ✅ 最流畅的viewer体验
- ✅ 最适合调试和演示
- ✅ FPS: ~50-60

---

### 多环境（批量测试）

```bash
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
    --viewer \
    --num-envs 4 \
    --device cuda
```

- ✅ 并行测试多个场景
- ✅ 提高数据收集效率
- ✅ FPS: ~40-50 (取决于GPU)

---

### 无头模式（最高性能）

```bash
python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
    --headless \
    --num-envs 16 \
    --device cuda
```

- ✅ 无渲染开销
- ✅ 最适合大规模测试
- ✅ FPS: ~200-500

---

## 🐛 常见问题

### Q1: USD文件找不到

```
❌ ERROR: USD file not found: third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd
```

**解决方案:**
```bash
# 检查文件是否存在
ls third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd

# 或使用绝对路径
--usd-path /home/ununtu/code/glab/genesislab/third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd
```

---

### Q2: AMO模型找不到

```
❌ ERROR: AMO model files not found
```

**解决方案:**
```bash
# 检查模型文件
ls data/AMO/amo_jit.pt
ls data/AMO/adapter_jit.pt
ls data/AMO/adapter_norm_stats.pt

# 或指定模型目录
--model-dir /path/to/your/amo/models
```

---

### Q3: Genesis初始化失败

```
❌ ERROR: Genesis backend not initialized
```

**解决方案:**
```bash
# 安装Genesis
pip install genesis-world

# 或更新到最新版本
pip install --upgrade genesis-world
```

---

### Q4: genPiHub导入错误

```
❌ ERROR: No module named 'genPiHub'
```

**解决方案:**
```bash
cd third_party/genPiHub
pip install -e .
```

---

### Q5: CUDA内存不足

```
❌ ERROR: CUDA out of memory
```

**解决方案:**
```bash
# 减少环境数量
--num-envs 1

# 或使用CPU
--device cpu
```

---

## 🔧 工作原理

### 架构流程

```
1. USD地形加载
   └─> TerrainCfg(terrain_type="usd", usd_path="Scene.usd")
   └─> 生成环境origins（网格布局）

2. AMO环境创建
   └─> AmoGenesisEnvCfg with USD terrain
   └─> 23 DOF G1人形机器人

3. AMO策略加载
   └─> 预训练的locomotion控制器
   └─> 50Hz控制频率

4. 主循环
   └─> 获取环境状态
   └─> 策略推理（observation → action）
   └─> 环境step
   └─> 渲染（如果启用viewer）
```

---

### 数据流

```
Keyboard Input → CommandState → Commands Array
                                      ↓
Environment State → Policy Observation → Policy
                                           ↓
                                    Joint Actions
                                           ↓
                                    Genesis Simulation
                                           ↓
                                    Updated State → Viewer
```

---

## 📂 相关文件

### 脚本文件
- [`play_amo_usd_terrain.py`](third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py) - 主脚本
- [`test_amo_usd_quick.sh`](third_party/genPiHub/scripts/amo/test_amo_usd_quick.sh) - 快速测试
- [`README_USD_TERRAIN.md`](third_party/genPiHub/scripts/amo/README_USD_TERRAIN.md) - 详细文档

### 核心实现
- [`scene_builder.py`](source/genesislab/genesislab/engine/scene/scene_builder.py) - USD地形实现
- [`amo_env_builder.py`](third_party/genPiHub/genPiHub/configs/amo_env_builder.py) - AMO环境配置

---

## 🎉 成功标准

你的设置成功如果：

- ✅ 脚本启动无错误
- ✅ Viewer打开并显示USD地形
- ✅ 人形机器人出现在地形上
- ✅ 机器人响应命令行走
- ✅ FPS稳定（单环境>30 fps）

---

## 💡 高级用法

### 1. 录制视频

使用Genesis viewer内置录制功能：
1. 启动viewer模式
2. 点击录制按钮
3. 运行脚本
4. 视频自动保存

---

### 2. 批量测试不同速度

```bash
for vx in 0.2 0.3 0.4 0.5; do
    python third_party/genPiHub/scripts/amo/play_amo_usd_terrain.py \
        --headless \
        --vx $vx \
        --max-steps 1000 \
        --print-every 100
done
```

---

### 3. 自定义地形工作流

1. **设计地形** - 使用Blender/Maya/Houdini
2. **导出USD** - 导出为.usd或.usda文件
3. **测试加载** - 使用USD地形测试脚本
4. **运行AMO** - 使用本脚本在地形上运行策略

---

## 🚀 下一步

完成基础测试后：

1. ✅ 尝试不同的USD地形
2. ✅ 实验不同的命令速度
3. ✅ 测试多环境场景
4. ✅ 集成到你的训练pipeline
5. ✅ 创建自定义地形

---

## 📞 获取帮助

- 📖 详细文档: [`scripts/README_USD_TERRAIN.md`](scripts/README_USD_TERRAIN.md)
- 🔧 USD地形实现: [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md)
- 🎮 AMO文档: [`third_party/genPiHub/scripts/amo/README_USD_TERRAIN.md`](third_party/genPiHub/scripts/amo/README_USD_TERRAIN.md)

---

**祝你在Scene.usd上运行AMO策略成功！** 🎉
