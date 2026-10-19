# py_inv

InSAR 同震断层滑动反演（Okada 弹性半空间模型）。

由 MATLAB 脚本移植而来，支持**多段断层**、**多道 InSAR / 方位向偏移（AZO）联合反演**，
以四叉树降采样 + 有界最小二乘求解断层面上的走滑 / 倾滑分布。

---

## 特性

- **多段断层**：断层迹线按行给出，每段可独立指定倾角（`##dip`）。
- **多道联合**：同一轨道共享一组 ramp，不同轨道各占一组 ramp 系数。
- **四叉树降采样**：按数据梯度自适应加密；可再按模型重采样做二次反演。
- **有界求解**：可施加走滑 / 倾滑的正负约束，配合 Tikhonov 一阶差分平滑与零滑动边界约束。
- **Okada 核可切换后端**：纯 NumPy 或 Numba（`##okada_backend`）。

---

## 目录结构

```
py_inv/
├── run_inversion.py          # 唯一命令行入口（改顶部 START / END 后运行）
│
├── geodetic/                 # 科学内核：纯计算，无副作用，可独立 import
│   ├── config.py             #   配置解析（configfile.txt / configpara.txt / data_list）
│   ├── coords.py             #   横轴墨卡托投影与 2-D 旋转
│   ├── solver.py             #   有界线性最小二乘（lsq_linear，兜底 least_squares）
│   ├── constraints.py        #   边界 lb/ub、零滑动边界、ramp 列
│   ├── okada.py              #   Okada 1985 位移核（NumPy / Numba）
│   ├── green.py              #   单道 Green 矩阵（LOS / AZO）
│   ├── green_function.py     #   读采样数据 + ramp 分发，组装 Green
│   ├── smoothing.py          #   Tikhonov 一阶差分平滑算子 H
│   ├── fault_geometry.py     #   断层几何 → 片元（patch）表
│   ├── insar_data.py         #   四叉树 / 均匀降采样，.grd 读写
│   ├── resample.py           #   按模型重采样（二次反演前）
│   ├── multilook.py          #   多视平均
│   ├── slip2insar.py         #   正演：滑移 → 合成 LOS
│   ├── slip2azo.py           #   正演：滑移 → 合成方位向偏移
│   ├── invert.py             #   反演主核 make_fault_from_insar1
│   ├── show_slip_model.py    #   3-D 滑移模型可视化
│   └── plot_insar_model_resampled.py   # 逐道 观测 / 模型 / 残差 三联图
│
├── workflows/                # 编排层：串联调用，不含科学计算
│   ├── inversion.py          #   主流程（四阶段调度）
│   ├── checkerboard.py       #   棋盘格分辨率测试
│   ├── dip_search.py         #   断层倾角网格搜索
│   ├── make_rms.py           #   平滑系数 λ 扫描（L-curve）
│   └── paragram.py           #   棋盘格测试的共享参数
│
├── configfile.txt            # 路径配置
├── configpara.txt            # 数值与反演参数
├── datav0.list               # 数据道清单
├── fault_trace.txt           # 断层迹线
├── seg_connect               # 段间平滑连接关系
│
├── matlab_code/              # 原始 MATLAB 参考实现
├── comsol_scripts/           # COMSOL 对照实验脚本（未接入主流程，见文末清单）
├── input_data/               # 观测数据
├── figures/                  # 图件
├── make_rms_out/             # λ 扫描产物
└── dip_search_out_1/         # 倾角搜索产物
```

依赖方向是单向的：**`run_inversion.py` → `workflows/` → `geodetic/`**。

---

## 安装

开发环境为 Python 3.13。

```bash
pip install numpy scipy            # 必需
pip install xarray                 # 读写 GMT netCDF .grd
pip install matplotlib             # 绘图
pip install numba                  # 可选：Okada 核加速（backend='numba'）
```

---

## 快速开始

1. 准备好 `configfile.txt`、`configpara.txt` 与各道数据（见下方「配置说明」「数据组织」）。
2. 编辑 `run_inversion.py` 顶部的阶段开关：

   ```python
   START = "invert1"
   END   = "invert1"
   ```

3. 运行：

   ```bash
   python run_inversion.py
   ```

---

## 工作流阶段

阶段名与 MATLAB 步骤一一对应，`run()` 会执行 `START` 到 `END` 之间的连续区间：

| 阶段 | MATLAB | 作用 | 主要产物 |
|---|---|---|---|
| `downsample` | Step 3 | 原始 `.grd` → 四叉树降采样 | `<track>/los_samp{iter_step}.mat` |
| `invert1` | Step 5 | 首次反演 | `py_inversion_iint0.mat` + `_show.png` |
| `resamp` | Step 6 | 用模型场重新采样 | `<track>/los_samp{iter_step2}.mat` |
| `invert2` | Step 7 | 二次反演 | `py_inversion_iint{iter_step2}.mat` + `_show_step7.png` |

完整流程：

```python
START = "downsample"
END   = "invert2"
```

> 阶段执行前**总会**先加载配置与断层几何；若各道已有 `los_samp{iter_step}.mat`，
> 可设 `SKIP_DOWNSAMPLE = True` 跳过 Step 3。

反演内核的数据流：

```
各道 los_samp{iint}.mat
      │
      ├─ green_function ── green ── okada      →  G（逐道 Green 矩阵，含 ramp 列）
      ├─ smoothing                             →  H（Tikhonov 一阶差分）
      ├─ constraints                           →  Wb / Wl / Wr（零滑动边界）、lb / ub
      │
      ▼
Greens = [ G ; H·λ/h1 ; Wb ; Wl ; Wr ]   →   min‖Greens·u − b‖  s.t. lb ≤ u ≤ ub
      │
      ▼
slip_model[:, 11:13]（走滑 / 倾滑）· RMS · Mw   →   .mat + PNG
```

---

## 配置说明

两个文件都用 `##key` 分段，`#` 开头的行视为注释。

### `configfile.txt` — 路径

| 键 | 说明 |
|---|---|
| `data_list` | 数据道清单文件（默认 `datav0.list`） |
| `fault_file` | 断层迹线文件（默认 `fault_trace.txt`） |
| `segment_file` | 段间平滑连接关系（默认 `seg_connect`） |

### `configpara.txt` — 数值与反演参数

| 键 | 说明 |
|---|---|
| `lonc` / `latc` | 局部坐标系原点（度） |
| `ref_lon` | UTM 中央经线，通常 ≤ `lonc` |
| `iter_step` | 首次反演的迭代标记（= 0 初始） |
| `iter_step2` | 重采样后二次反演的迭代标记 |
| `con` | 约束开关：走滑 / 倾滑 / 法向（正 = 下界归零，负 = 上界归零，0 = 无约束） |
| `dip` | 各段倾角（个数须等于断层段数） |
| `ramp_choice` | ramp 类型：`none` / `bi_ramp`(4) / `qu_ramp_5`(5) / `qu_ramp_7`(7) |
| `w_ratio` / `l_ratio` | 片元沿倾向 / 走向的尺寸递增比 |
| `width` | 断层宽度（m） |
| `len_top` | 顶部片元长度（m） |
| `layers` | 倾向分层数 |
| `default_region` | 数据清单缺省采样框 `xmin xmax ymin ymax` |
| `okada_backend` | `numpy` / `numba` / `auto` |
| `max_nfev` | 求解器最大迭代次数 |
| `model_type` | `okada`（分层 EDCMP 模型未移植） |

### `datav0.list` — 数据道清单

每行一道：

```
<track_path>  <npt>  [xmin xmax ymin ymax]  <type>
```

- `<type>` 取 `insar`（或 `rng`）或 `azo`；
- 只给 3 列时使用 `##default_region`；
- 同一轨道号（路径中的 `A143` / `D106` 等）共享一组 ramp 系数。

### `fault_trace.txt` — 断层迹线

每行一段：`lon1  lat1  lon2  lat2`。

### `seg_connect` — 段间连接

每行描述相邻两段的端部连接：`<段号1> left <段号2> right`。

---

## 数据组织

每道一个目录，`downsample`（四叉树）需要：

```
input_data/data_v5_opt1/D106/
├── los_clean_detrend.grd     # 解缠并去趋势后的 LOS（主输入）
├── dem.grd                   # 地形
├── look_e.grd                # 视线方向东分量
├── look_n.grd                # 视线方向北分量
└── look_u.grd                # 视线方向垂直分量
```

`downsample` 会额外写出 `dem_low.grd`、`look_{e,n,u}_low.grd` 与 `los_samp{iter_step}.mat`；
`resamp` 直接复用这些 `*_low.grd`。

> 若只需反演（跳过 Step 3），每道至少要有 `los_samp{iint}.mat`。

---

## 输出产物

| 文件 | 内容 |
|---|---|
| `py_inversion_iint0.mat` | 首次反演快照：`slip_model`、`u`、`G_last`、`GrF`、`H`、`Wb/Wl/Wr`、`bdata_sm`、`RMS_misfit`、`model_roughness`、`return_var` 等 |
| `py_inversion_iint1.mat` | 二次反演快照（结构同上） |
| `*_show.png` | 滑移模型三维图（叠断层迹线） |
| `<track>/los_samp{iint}_misfit_iter{iint}.png` | 逐道 观测 / 模型 / 残差 三联图 |
| `<track>/los_model.mat` | 该道的模型场 |

`.mat` 与 `.npy` 已在 `.gitignore` 中忽略。

---

## 命令行工具

```bash
python run_inversion.py                 # 主流程
python -m workflows.make_rms            # 平滑系数 λ 扫描（L-curve）→ make_rms_out/
python -m workflows.dip_search          # 断层倾角网格搜索 → dip_search_out/
python -m workflows.checkerboard        # 棋盘格分辨率测试（Okada 正演 → 反演还原）
```

各脚本均支持 `--help`；`make_rms` / `dip_search` 的 `--config-dir`、`--out-dir` 默认指向项目根。

也可以直接当库用：

```python
from geodetic import load_fault_one_plane, make_fault_from_insar1
from workflows.inversion import run, InversionWorkflowState

run(InversionWorkflowState(), start="invert1", end="invert1")        # 全流程

slip_vs = load_fault_one_plane("fault_trace.txt", dip=[82, 82, 82, 82],
                               lonc=95.33, latc=19.61, ref_lon=95.0)  # 单步反演
slip, rms, rough, ret, extras = make_fault_from_insar1(
    slip_vs, None, 0, tracks, paths_type=["azo"] * 4, ...)
```

---

## 与 MATLAB 版本的对应

`matlab_code/` 保留原始 MATLAB 实现，Python 侧逐函数对应：

| MATLAB | Python |
|---|---|
| `InversionExample.m` | `workflows/inversion.py` |
| `load_fault_one_plane.m` | `geodetic/fault_geometry.py` |
| `calc_okada.m` | `geodetic/okada.py` |
| `calc_green_{insar,AZO}_okada.m` | `geodetic/green.py` |
| `build_green_function.m` | `geodetic/green_function.py` |
| `build_smooth_function.m` | `geodetic/smoothing.py` |
| `bounds_new.m` / `zero_slip_boundary.m` | `geodetic/constraints.py` |
| `make_insar_data.m` / `make_insar_downsample.m` | `geodetic/insar_data.py` |
| `resamp_insar_data.m` | `geodetic/resample.py` |
| `slip2{insar,AZO}_okada.m` | `geodetic/slip2insar.py` / `geodetic/slip2azo.py` |
| `make_fault_from_insar1.m` | `geodetic/invert.py` |
| `show_slip_model.m` | `geodetic/show_slip_model.py` |
| `make_rms.m` | `workflows/make_rms.py` |
| `lsqlin` | `geodetic/solver.py`（`scipy.optimize.lsq_linear`） |

---

## 说明

- 反演对输入完全确定：相同配置与数据可复现逐位一致的结果。

### `comsol_scripts/` 脚本清单

早期用于把 Okada 解析解与 COMSOL 有限元结果做交叉验证，**未接入主流程**。
均需从项目根目录运行（脚本已自动把项目根加入 `sys.path`）：

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `comsol_green.py` | 棋盘格：逐片元驱动 COMSOL 求解单位滑移场，组装 `Green_comsol.npy` | COMSOL + `top{0,1}_*.mph` |
| `analysis_comsol.py` | 对比 `Green_okada.npy` / `Green_comsol.npy`：整体与逐片元相对误差、方向余弦，并把失配残差投影到走滑 / 倾滑子空间 | 两个 `.npy` |
| `comsol_self_consistent.py` | COMSOL 自洽测试：用 COMSOL Green 自造合成观测再反演还原 | `Green_comsol.npy` |
| `Green_diff_copy.py` | 交互浏览逐片元 Okada vs COMSOL 位移场（3×3 子图，可切换 patch） | 两个 `.npy` |
| `okada_vs_comsol.py` | 单 patch：Okada vs 多个 `.mph` 模型，并排 + 交互 A/B 对比 | COMSOL |
| `okada_vs_comsol_copy.py` | 单 patch：Okada vs 单个 `.mph`，3×3 散点对比（早期版本） | COMSOL |
| `comsol_gui_vs_java.py` | 与 `v.txt`（COMSOL GUI 实测）对比，含 A/B 下拉选择 | COMSOL + `v.txt` |
| `comsol_vs_comsol.py` | 同一模型下 uniform / taper1d / taper2d × `excludedEdges` 的边界条件组合对比 | COMSOL |
| `comsol_slip_smooth.py` | 余弦窗（raised-cosine taper）形状演示 | — |
| `export_okada_vs_comsol_png.py` | 论文用高分辨率 PNG 导出（按 `s` 导出当前视野） | COMSOL |
| `comsol_run.py` | 从反演工作流状态出发，生成缅甸断层的 COMSOL Green 矩阵（`Green_comsol*.mat`） | COMSOL + 采样数据 |
| `comsol_result.py` | 用 COMSOL Green 替换 Step 5 快照并重新反演；默认与快照（Okada）生成 `A / B / A−B` 三联对比图，可用 `--compare` 追加其它模型 | `Green_comsolGD.mat` + 快照 |

`comsol_result.py` 的对比对象可以任意指定（默认对比 Step 5 快照里的 Okada 结果）：

```bash
# 只与 Okada 对比
python comsol_scripts/comsol_result.py --comsol-mat Green_comsolGD.mat

# 再追加一次其它 COMSOL 反演结果作为对比（--compare 可重复）
python comsol_scripts/comsol_result.py --comsol-mat Green_comsolGD.mat \
    --compare "COMSOL_v0=mphs/v0/comsol_inversion_out/comsol_inversion.mat"

# 只看二维对比图，不生成 3D
python comsol_scripts/comsol_result.py --comsol-mat Green_comsolGD.mat --no-3d
```

产物（写入 `--out-dir`，默认 `comsol_inversion_out/`）：

| 文件 | 内容 |
|---|---|
| `slip3d_<模型>.png` | 每个模型一张 3D 滑移图，**所有模型共用同一色标**，可直接横向比较 |
| `compare_<模型>.png` | `A / B / A−B` 三联图：行为走滑 / 倾滑 / 滑移量，列为两个模型与差值；断层面展开为「沿走向里程 — 深度」平面，A/B 共用色标，差值用对称色标 |
| `slip_comparison_stats.csv` | 各模型的 RMS、Mw、M0、最大 / 平均滑移，以及与主模型的 RMS 差、max\|Δ\|、相对 L2、相关系数 |

## 许可

见 [LICENSE](LICENSE)。
