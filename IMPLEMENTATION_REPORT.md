# See-Shell OCT 点云可视化系统 — 实现报告

## 1. 数据概况

### 1.1 数据源

`oct_cloud/` 目录下共有 153 个子目录，其中包含有效体素数据的目录可分为两类：

| 类型 | 目录数 | 有数据的目录 | 说明 |
|------|--------|-------------|------|
| `scan_array_*` | 5 | 2 | 多位置网格扫描（3×3、5×5 等），每个位置生成一个 PLY/NPY 文件 |
| `scan_session_*` | 141 | 9 | 单位置扫描，每个目录包含一个 PLY/NPY 文件 |
| 其他（calibration、accel_sweep_test 等） | 6 | 0 | 测试/校准数据，无 PLY |

有效数据文件共计 **39 个 PLY** + **38 个 NPY**（其中一个 PLY 无对应 NPY）。

### 1.2 数据格式

#### NPY 文件（原始体素数据）

```
shape: (500000, 4)
dtype: float32
列定义: [x, y, z, intensity]
坐标范围: x ∈ [0, 511], y ∈ [0, 511], z ∈ [0, 511]
强度范围: [0.358, 1.000]
```

每个 NPY 包含 500K 个非零体素采样点，空间分布在 512³ 的规则网格上（总体素量约 1.34 亿）。数据本质是三维 OCT 体数据在稀疏非零位置上的采样。

#### PLY 文件（预着色点云）

```
点数: 500,000
属性: points (x,y,z) + RGB (uint8)
RGB 唯一色值: 255 种（量化后的 colormap 产物）
RGB 范围: [1, 254]
坐标范围: 与 NPY 相同，x/y/z ∈ [0, 511]
```

PLY 文件是 NPY 的彩色版本——原始 intensity 经某种 colormap（非标准 jet/viridis）映射为 RGB。每个 PLY 约 13.5 MB。

### 1.3 扫描网格布局

- `scan_array_20260324_153718`：3×3 网格，9 个位置，位置标签为 `n1/n1` 到 `p1/p1`
- `scan_array_20260324_235201`：5×5 网格，20 个位置，位置标签从 `n2/n2` 到 `p2/p2`
- `scan_session_*`：单位置扫描，共 9 个有效文件

## 2. 技术架构

### 2.1 技术选型与演进

| 方案 | 阶段 | 结果 |
|------|------|------|
| Three.js / 浏览器 | 初步构想 | 否决——500K 点有浏览器崩溃风险 |
| Matplotlib 3D | 第一版实现 | 否决——CPU 渲染，交互卡顿严重 |
| Open3D + Jupyter | 尝试 | 否决——Mac 上 GUI 不稳定 |
| PyVista 体渲染 (VTK GPU volume rendering) | 完整实现（727 行） | 否决——用户反馈过于复杂，且与原始数据量级不匹配 |
| **PyVista + PyQt6 点云渲染** | **最终方案** | **采纳**——直接渲染 PLY 文件，稳定、高性能 |

最终选型理由：
- VTK (9.6.1) 底层使用 OpenGL 渲染点云，Mac Pro 上 500K 点交互流畅
- `pyvistaqt.QtInteractor` 提供嵌入 Qt 的 3D 视口，支持旋转/缩放/平移
- PyQt6 提供原生 GUI 控件（slider、combo box、list widget），响应延迟 < 1ms
- 通过 `uv` 管理依赖，环境一致性好

### 2.2 依赖管理

```toml
# pyproject.toml
dependencies = [
    "numpy>=2.0",
    "pyvista>=0.43",
    "pyvistaqt>=0.11",
    "PyQt6>=6.5",
    "vtk>=9.2",
]
```

运行方式：`uv run python ply_viewer.py`，无需手动激活虚拟环境。

## 3. 核心实现

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│  Viewer (QMainWindow)                               │
│  ┌──────────┐  ┌──────────────────────────────────┐ │
│  │ Gallery  │  │  QtInteractor (VTK 3D viewport)  │ │
│  │          │  │                                    │ │
│  │ thumbnail│  │  500K points, rgb=True, spheres   │ │
│  │ list     │  │                                    │ │
│  │ (39项)   │  ├──────────────────────────────────┤ │
│  │          │  │ Controls:                         │ │
│  │ QThread  │  │  Row 1: info | point size | BG    │ │
│  │ lazy     │  │  Row 2: Z min─── Z max─── reset  │ │
│  │ load     │  │  Row 3: X min─── X max─── reset  │ │
│  │          │  │  Row 4: Y min─── Y max─── reset  │ │
│  │          │  │  Row 5: filter mode | min | max   │ │
│  │          │  │  Row 6: export params             │ │
│  └──────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 3.2 缩略图生成（MIP 投影）

缩略图采用 Maximum Intensity Projection 从 NPY 体素数据生成二维预览图，而非从 3D 渲染截图：

```python
def _mip_thumbnail(npy_path):
    npy = np.load(npy_path)                    # (500000, 4)
    img = np.zeros((512, 512), dtype=np.float32)
    x = npy[:, 0].astype(int)
    y = npy[:, 1].astype(int)
    np.maximum.at(img, (x, y), npy[:, 3])      # 沿 z 轴取最大强度
    img = img / img.max() * 255                 # 归一化到 0-255
    return QImage(img, 512, 512, ...)           # 转为 Qt 图像
```

关键设计决策：
- 不分配 512³ 的完整体数据（约 268MB float32），而是直接在 512×512 的 2D 累积数组上操作
- `np.maximum.at` 是 unbuffered 操作，对重复坐标取最大值而非覆盖
- 单张缩略图生成耗时约 5ms（纯 NumPy，无 GPU 依赖）

### 3.3 异步加载

缩略图在 `QThread` 中后台生成，避免阻塞 GUI 主线程：

```python
class ThumbnailLoader(QThread):
    done = pyqtSignal(int, QPixmap)

    def run(self):
        for i, scan in enumerate(self.scans):
            pixmap = generate_thumbnail(scan)
            self.done.emit(i, pixmap)    # 跨线程信号，安全的 Qt 机制
```

- 39 张缩略图在约 200ms 内全部生成完毕
- 通过 `pyqtSignal` 将 QPixmap 传回主线程，线程安全
- 主窗口启动时即启动后台线程，用户无需等待

### 3.4 颜色/强度筛选

支持四种筛选模式，每种将 RGB 颜色映射为 0-255 的标量度量值：

| 模式 | 计算方法 | 用途 |
|------|---------|------|
| Intensity | `0.299R + 0.587G + 0.114B` | 灰度强度，最常用 |
| Hue (warm↔cool) | HSV 色相，0°/255 对应冷色/暖色 | 按色温筛选 |
| Brightness | `max(R, G, B)` | 最亮通道值 |
| Saturation | HSV 饱和度 | 区分彩色/灰色区域 |

HSV 转换使用**纯 NumPy 向量化实现**（非 `colorsys` 逐点循环），对 500K 点一次计算耗时 < 10ms：

```python
def _rgb_to_hsv_vec(rgb):
    # 50 万个点并行计算，无 Python 循环
    hue[mr] = 60.0 * (((g[mr] - b[mr]) / delta[mr]) % 6)
    ...
```

用户通过 Min/Max 双滑块（0-255 范围）控制可见点范围。筛选结果实时显示为 `"342,000/500,000 (68%)"` 格式。

### 3.5 空间范围裁剪

X/Y/Z 三轴各有独立的 Min/Max 滑块，用于空间裁剪：

- 内部滑块范围固定为 0-1000（整数精度）
- 通过线性映射转换到实际坐标值：`val = axis_min + (axis_max - axis_min) * slider / 1000`
- 每次加载新扫描时自动计算该扫描的轴范围边界
- 标签实时显示实际坐标值（如 `Z max: 511.0`）

空间裁剪与颜色筛选使用 **AND 逻辑**组合——只显示同时满足所有条件的点。

### 3.6 点云渲染管线

```
原始数据 (orig_pts, orig_rgb)
        │
        ▼
构建 boolean mask（颜色筛选 AND 空间裁剪）
        │
        ▼
mask 过滤: pts = orig_pts[mask], rgb = orig_rgb[mask]
        │
        ▼
pv.PolyData(pts) + cloud["RGB"] = rgb
        │
        ▼
plotter.add_mesh(cloud, rgb=True,
                 point_size=N,
                 render_points_as_spheres=True)
        │
        ▼
VTK OpenGL 渲染 → Mac GPU 加速
```

关键性能特征：
- 每次筛选变化时**重建整个 PolyData**（而非修改已有 mesh），避免 VTK 的增量更新开销
- `render_points_as_spheres=True` 使点在视觉上更清晰
- 点大小通过 `actor.GetProperty().SetPointSize()` 实时调整，无需重建 mesh
- 500K 点的 mask 计算耗时 < 5ms（NumPy 向量化），重建 + 渲染约 50ms

### 3.7 参数导出

点击 "Export Params" 按钮将当前所有可视化参数保存为 JSON：

```json
{
  "scan_file": "/path/to/volume_pointcloud_...ply",
  "scan_group": "20260324 153718",
  "scan_suffix": "0_0",
  "point_size": 3,
  "background_color": "#0D0D14",
  "filter_mode": "Intensity",
  "filter_min": 0,
  "filter_max": 255,
  "x_min": 0.0,
  "x_max": 511.0,
  "y_min": 0.0,
  "y_max": 511.0,
  "z_min": 128.3,
  "z_max": 384.7
}
```

导出的坐标值是实际空间坐标（非滑块内部值），可直接用于后续分析流水线。

## 4. 性能指标

| 操作 | 耗时 | 备注 |
|------|------|------|
| 单张 MIP 缩略图 | ~5ms | NumPy, 512×2 累积 |
| 39 张缩略图全量生成 | ~200ms | QThread 后台，不阻塞 GUI |
| 加载一个 PLY 文件 | ~300ms | pyvista.read, 13.5MB |
| 颜色 metrics 预计算 (500K pts) | ~10ms | 4 种模式一次性算完 |
| 滑块触发重建 mesh | ~50ms | mask + PolyData + render |
| 点大小调整 | <5ms | 仅修改 actor property |
| 内存占用 (单扫描) | ~30MB | pts + rgb + metrics |

## 5. 文件清单

| 文件 | 行数 | 用途 |
|------|------|------|
| `ply_viewer.py` | 517 | 主程序，点云浏览器 |
| `pyproject.toml` | — | uv 项目配置 + 依赖声明 |
| `oct_cloud/` | — | 数据目录，39 个有效 PLY + 38 个 NPY |
| `oct_viewer.py` | 727 | 早期体渲染方案（已弃用，保留参考） |

## 6. 已知局限

1. **NPY 缺失**: 39 个 PLY 中有 1 个（`p1_n2`）无对应 NPY，缩略图退化为一维灰色条
2. **筛选无 alpha 通道**: 当前只支持二值筛选（显示/隐藏），不支持半透明渐变
3. **每次重建完整 mesh**: 空间裁剪和颜色筛选变化时重建全部 PolyData，未做增量更新
4. **无多选对比**: 一次只能查看一个扫描，不支持并排对比两个位置的数据
5. **相机状态丢失**: 切换扫描时 `reset_camera()`，用户的手动视角不会保留
