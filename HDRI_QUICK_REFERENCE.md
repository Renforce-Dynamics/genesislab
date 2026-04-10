# HDRI 环境光 - 快速参考

## ⚠️ 依赖要求

**HDRI 需要 LuisaRender（Genesis RayTracer 子模块）**

⚠️ **不能**通过 pip 安装！需要从 Genesis 源码编译。

详细安装说明见: [HDRI_SETUP_GUIDE.md](HDRI_SETUP_GUIDE.md#系统要求)

**没安装？** 系统会自动使用方向光降级，不影响运行。

## 🎯 最简单的使用方法

```python
from genesislab.engine.scene import SceneCfg
from genesislab.utils import create_vis_options_with_hdri

# 一行代码启用 HDRI 环境光
scene_cfg = SceneCfg(
    viewer=True,
    vis_options=create_vis_options_with_hdri(),  # 自动配置 HDRI
)
```

## 📁 HDRI 文件位置

```
genesislab/data/assets/hdri/sky.hdr
```

## 🔧 手动配置

```python
from genesislab.engine.sim import VisOptionsCfg

vis_options = VisOptionsCfg(
    env_surface="sky.hdr",      # HDRI 文件名
    env_radius=1000.0,           # 环境球半径
    env_pos=(0.0, 0.0, 0.0),    # 环境球位置
)
```

## 📦 下载 HDRI 文件

免费高质量 HDRI：
- **Poly Haven**: https://polyhaven.com/hdris
- **HDRI Haven**: https://hdrihaven.com/

推荐下载：
- `kloppenheim_06_puresky_4k.hdr` - 纯净天空
- `cloud_layers_4k.hdr` - 云层
- `studio_small_09_4k.hdr` - 工作室照明

## 🛠️ 检查配置

```bash
# 运行测试脚本
python3 /tmp/test_hdri_check.py
```

或者在 Python 中：

```python
from genesislab.utils import hdri_exists, get_default_hdri_path

# 检查文件是否存在
if hdri_exists("sky.hdr"):
    print("✅ HDRI ready!")
else:
    print(f"❌ Place HDRI at: {get_default_hdri_path()}")
```

## 📚 完整文档

详细说明请参考：`HDRI_SETUP_GUIDE.md`
示例代码：`examples/hdri_lighting_example.py`

---

**💡 提示**: 如果没有 HDRI 文件，系统会自动使用方向光照明，不会报错。
