# HDRI Environment Lighting Setup Guide

本指南说明如何在 GenesisLab 中使用 HDRI 环境光（IBL - Image-Based Lighting）。

## 功能概述

新增了 HDRI 环境光支持，可以为场景提供真实的全局照明和反射。主要特性：

- ✅ 支持 `.hdr` 格式的 HDRI 文件
- ✅ 自动路径解析（相对路径或绝对路径）
- ✅ 可配置的环境球半径和位置
- ✅ 兼容 RayTracer 渲染器
- ✅ 提供备用的方向光配置
- ✅ 便捷的辅助函数

## 系统要求

### 必需依赖

HDRI 环境光需要 **LuisaRender**（Genesis 的 RayTracer 渲染器）。

**重要**: LuisaRender **不能**通过 pip 安装！它是 Genesis 的一个可选子模块，需要从源码编译。

#### 安装 LuisaRender（高级用户）

如果你是从源码安装的 Genesis，可以按以下步骤构建 LuisaRender：

```bash
cd /path/to/Genesis/genesis/ext/LuisaRender

# 初始化子模块
git submodule update --init --recursive

# 构建（需要 CMake 3.18+, CUDA）
mkdir -p build
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D PYTHON_VERSIONS=3.11 \
    -D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON \
    -D LUISA_COMPUTE_DOWNLOAD_OIDN=ON \
    -D LUISA_COMPUTE_ENABLE_GUI=OFF \
    -D LUISA_COMPUTE_ENABLE_CUDA=ON \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake --build build -j $(nproc)
```

**依赖要求**:
- CMake 3.18 或更高版本
- CUDA 工具链
- 充足的编译时间（可能需要 10-30 分钟）

#### 使用 Docker（推荐）

如果需要 HDRI 功能，最简单的方法是使用 Genesis 官方 Docker 镜像：

```bash
cd /path/to/Genesis
docker build -t genesis -f docker/Dockerfile docker
```

**注意**: 如果未安装 LuisaRender，系统会自动降级使用方向光，不会报错。

## 快速开始

### 1. 准备 HDRI 文件

将你的 `sky.hdr` 文件放置到以下位置：

```
genesislab/data/assets/hdri/sky.hdr
```

**获取免费 HDRI 资源：**
- [Poly Haven](https://polyhaven.com/hdris) - 高质量免费 HDRI
- [HDRI Haven](https://hdrihaven.com/) - 免费全景 HDR
- [sIBL Archive](http://www.hdrlabs.com/sibl/archive.html) - Smart IBL 存档

### 2. 方法一：使用便捷函数（推荐）

```python
from genesislab.engine.scene import SceneCfg
from genesislab.utils import create_vis_options_with_hdri

# 自动配置 HDRI 环境光
scene_cfg = SceneCfg(
    num_envs=32,
    viewer=True,
    vis_options=create_vis_options_with_hdri(
        hdri_name="sky.hdr",  # HDRI 文件名（可选，默认为 sky.hdr）
        env_radius=1000.0,    # 环境球半径（可选）
        add_lights=True,      # 添加备用方向光（可选）
    ),
    # ... 其他配置
)
```

### 3. 方法二：手动配置

```python
from genesislab.engine.scene import SceneCfg
from genesislab.engine.sim import VisOptionsCfg

scene_cfg = SceneCfg(
    num_envs=32,
    viewer=True,
    vis_options=VisOptionsCfg(
        # HDRI 环境光配置
        env_surface="sky.hdr",      # 相对于 data/assets/hdri/ 的路径
        env_radius=1000.0,           # 环境球半径
        env_pos=(0.0, 0.0, 0.0),    # 环境球位置
        
        # 可选：添加方向光作为补充
        lights=[
            {
                "type": "directional",
                "dir": (-1.0, -1.0, -2.0),
                "color": (1.0, 1.0, 0.95),
                "intensity": 5.0,
            },
        ],
        
        # 可选：环境光和背景色
        ambient_light=(0.3, 0.3, 0.3),
        background_color=(0.5, 0.7, 1.0),
    ),
    # ... 其他配置
)
```

### 4. 方法三：使用工作室照明（无需 HDRI）

如果没有 HDRI 文件，可以使用预设的工作室照明：

```python
from genesislab.utils import create_studio_lighting

scene_cfg = SceneCfg(
    vis_options=create_studio_lighting(
        key_intensity=5.0,   # 主光强度
        fill_intensity=2.0,  # 补光强度
    ),
)
```

## 配置选项详解

### VisOptionsCfg 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `env_surface` | str | None | HDRI 文件路径（相对或绝对） |
| `env_radius` | float | 1000.0 | 环境球半径 |
| `env_pos` | tuple | (0, 0, 0) | 环境球位置 |
| `lights` | list[dict] | None | 光源列表配置 |
| `ambient_light` | tuple | None | 环境光颜色 (r, g, b) |
| `background_color` | tuple | None | 背景颜色 (r, g, b) |

### 光源配置格式

方向光（Directional Light）：
```python
{
    "type": "directional",
    "dir": (x, y, z),           # 光线方向
    "color": (r, g, b),         # 光源颜色 [0-1]
    "intensity": float,         # 强度
}
```

点光源（Point Light）：
```python
{
    "type": "point",
    "pos": (x, y, z),           # 光源位置
    "color": (r, g, b),         # 光源颜色 [0-1]
    "intensity": float,         # 强度
}
```

## 代码修改详情

### 修改的文件

1. **`source/genesislab/genesislab/engine/sim/sim_cfg.py`**
   - 扩展 `VisOptionsCfg` 类，添加 HDRI 和光源配置参数

2. **`source/genesislab/genesislab/engine/scene/scene_builder.py`**
   - 添加 `_setup_environment_lighting()` 方法
   - 自动配置 HDRI 环境光和 RayTracer 渲染器

3. **`source/genesislab/genesislab/engine/scene/lab_scene_cfg.py`**
   - 更新文档说明如何使用 HDRI 环境光

4. **`source/genesislab/genesislab/utils/lighting.py`** (新增)
   - 提供便捷函数用于快速配置照明

5. **`source/genesislab/genesislab/utils/__init__.py`** (新增)
   - 导出照明相关函数

### 新增的辅助函数

- `create_vis_options_with_hdri()` - 创建带 HDRI 的可视化配置
- `create_studio_lighting()` - 创建工作室照明配置
- `hdri_exists()` - 检查 HDRI 文件是否存在
- `get_default_hdri_path()` - 获取默认 HDRI 路径

## 示例代码

查看完整示例：
```bash
python examples/hdri_lighting_example.py
```

## 注意事项

1. **渲染器兼容性**
   - HDRI 环境光需要 RayTracer 渲染器
   - 实时查看器会使用光栅化近似渲染
   - 建议同时配置方向光以获得更好的实时效果

2. **性能考虑**
   - HDRI 文件可能较大，建议使用 2K-4K 分辨率
   - 高分辨率 HDRI 会增加渲染时间

3. **文件格式**
   - 支持 `.hdr` (Radiance HDR) 格式
   - HDRI 应使用等距柱状投影（equirectangular projection）

4. **路径解析**
   - 相对路径会相对于 `data/assets/hdri/` 目录解析
   - 也可以使用绝对路径指定任意位置的 HDRI 文件

## 故障排除

### HDRI 未生效
- 检查文件路径是否正确
- 确认 `sky.hdr` 文件存在于 `data/assets/hdri/` 目录
- 查看日志输出，确认是否有警告信息

### 渲染效果不佳
- 尝试调整 `env_radius` 参数
- 添加额外的方向光补充照明
- 调整 `ambient_light` 参数增加环境亮度

### 文件未找到错误
```python
# 使用辅助函数检查文件是否存在
from genesislab.utils import hdri_exists, get_default_hdri_path

if not hdri_exists("sky.hdr"):
    print(f"HDRI file not found at: {get_default_hdri_path()}")
```

## 更多信息

- Genesis 文档: [Genesis Documentation](https://genesis-world.readthedocs.io/)
- HDRI 教程: 参考 `data/assets/hdri/README.md`
- 示例代码: `examples/hdri_lighting_example.py`
