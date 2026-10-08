r"""Port of ``make_rms.m`` — 平滑系数 λ 扫描 (对应工作流 Step 7 / ``iter_step2`` 反演).

对应 MATLAB 脚本的 λ 扫描主段:

  输入: 反演快照 ``.mat`` (默认 ``py_inversion_iint1.mat``, 即 Python 工作流 Step 7 输出;
        对应 MATLAB 中 ``load greens_cache*.mat`` / 工作区继承的变量), 需含
        ``G_last, GrF, H, Wb, Wl, Wr, h1, bdata_sm, Bdata, bd_last,
        slip_model, ramp_choice`` (可选 ``class_map, u``).

  对每个 λ 重建方程并求解 (与 ``make_fault_from_insar1`` 同一套求解链)::

      Greens = [G_last; H*λ/h1; Wb; Wl; Wr]      bdata_sm 不变
      min ||Greens*u - bdata_sm||²  s.t.  lb <= u <= ub   (scipy lsq_linear, TRF)

  输出 (默认 ``make_rms_out/``):
    1. ``scan_table.csv``                 — 每个 λ 的诊断量表 (RMS/redu%/粗糙度/Mw/最大滑移)
    2. ``rms_vs_lambda.png``              — RMS 残差与方差降低率随 log10(λ) 曲线 (.m 的两张 figure)
    3. ``lcurve.png``                     — 模型粗糙度–数据 misfit 权衡曲线
    4. ``slip_lam<λ>.png``                — 每个 λ 的 3D 滑移模型 (show_slip_model, 叠断层迹线)
    5. ``lam<λ>/track_<名>_misfit.png``   — 指定 λ 的逐轨道 观测/模型/残差 三联图 (plot_insar_model_resampled)
    6. ``lam<λ>/tracks_residual.png``     — 该 λ 的逐轨道残差汇总图 (叠断层迹线, 统一色标)

说明:
  * 逐轨道差异图需要各道 ``los_samp<iint>.mat`` (Step 7 采样点); 各道行数之和必须等于
    快照的 GrF 行数, 否则跳过绘图并给出提示 (λ 扫描本身只需快照, 不受影响).
  * make_rms.m 的 "strike-only" 试验段 (删除倾滑列再反演) 未移植.
  * ``mu`` 默认 30e9 (与 make_fault_from_insar1 一致; .m 的 compute_moment 用 33e9).

用法::

    python make_rms.py                                  # 默认快照 + 默认 λ 组
    python make_rms.py --lambdas 0.2,0.5,1.0,2.0,5.0    # 自定义 λ 组
    python make_rms.py --snapshot py_inversion_iint0.mat --iint 0
    python make_rms.py --no-slip-figs --misfit-lambdas 1.0
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares, lsq_linear

from bounds_new import bounds_new
from make_fault_from_insar1 import _add_col_of

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# 快照加载
# ---------------------------------------------------------------------------

_NUMERIC_VARS = (
    "G_last", "GrF", "H", "Wb", "Wl", "Wr", "h1",
    "bdata_sm", "Bdata", "bd_last", "slip_model",
)


def load_snapshot(mat_path: str) -> Dict[str, Any]:
    """读反演快照并做维度自检; 对应 MATLAB ``load greens_cache*.mat``."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError("快照不存在: %s" % mat_path)
    d = loadmat(mat_path)
    missing = [k for k in (*_NUMERIC_VARS, "ramp_choice") if k not in d]
    if missing:
        raise KeyError("%s 缺少变量: %s" % (mat_path, ", ".join(missing)))

    snap: Dict[str, Any] = {k: np.asarray(d[k], dtype=np.float64) for k in _NUMERIC_VARS}
    snap["ramp_choice"] = str(np.asarray(d["ramp_choice"]).ravel()[0])
    snap["h1"] = float(np.asarray(d["h1"]).ravel()[0])
    snap["class_map"] = (np.asarray(d["class_map"], dtype=int).ravel()
                         if "class_map" in d else np.ones(1, dtype=int))
    if "u" in d:
        snap["u"] = np.asarray(d["u"], dtype=np.float64).ravel()

    n_cols = snap["G_last"].shape[1]
    for name in ("GrF", "H", "Wb", "Wl", "Wr"):
        if snap[name].shape[1] != n_cols:
            raise ValueError(
                "快照列数不一致: %s 有 %d 列, G_last 有 %d 列"
                % (name, snap[name].shape[1], n_cols)
            )
    n_rows = (snap["G_last"].shape[0] + snap["H"].shape[0]
              + snap["Wb"].shape[0] + snap["Wl"].shape[0] + snap["Wr"].shape[0])
    if snap["bdata_sm"].shape[0] != n_rows:
        raise ValueError(
            "bdata_sm 行数 %d 与 [G_last;H;Wb;Wl;Wr] 行数 %d 不一致"
            % (snap["bdata_sm"].shape[0], n_rows)
        )
    return snap


# ---------------------------------------------------------------------------
# 求解 (与 make_fault_from_insar1 同一套链: lsq_linear exact→lsmr, 兜底 least_squares)
# ---------------------------------------------------------------------------

def _solve_bounded_lsq(
    Greens: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_nfev: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, str]:
    Greens_ = np.ascontiguousarray(Greens, dtype=np.float64)
    b_ = np.ascontiguousarray(b, dtype=np.float64).ravel()
    lb_ = np.ascontiguousarray(lb, dtype=np.float64).ravel()
    ub_ = np.ascontiguousarray(ub, dtype=np.float64).ravel()

    for solver in ("exact", "lsmr"):
        try:
            res = lsq_linear(
                Greens_, b_, bounds=(lb_, ub_), method="trf",
                tol=1e-12, lsq_solver=solver, max_iter=int(max_nfev), verbose=0,
                lsmr_tol=1e-12, lsmr_maxiter=min(10_000, 50_000 * Greens_.shape[1]),
            )
            if res.success or res.status in (1, 2, 3, 4):
                return (np.asarray(res.x, dtype=np.float64).ravel(),
                        "lsq_linear(%s) nit=%s %s" % (solver, res.nit, res.message))
        except (np.linalg.LinAlgError, ValueError, RuntimeError, MemoryError) as e:
            if verbose:
                print("  lsq_linear (lsq_solver=%s) failed: %r" % (solver, e), flush=True)

    if verbose:
        print("  lsq_linear 不可用 — 兜底 least_squares(TRF)", flush=True)
    x0 = np.clip(np.zeros(Greens_.shape[1]), lb_, ub_)
    res = least_squares(
        lambda x: Greens_ @ x - b_, x0, jac=lambda x: Greens_,
        bounds=(lb_, ub_), method="trf", tr_solver="lsmr",
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=max(500, int(max_nfev) * 20), verbose=0,
    )
    return (np.asarray(res.x, dtype=np.float64).ravel(),
            "least_squares success=%s %s" % (res.success, res.message))


# ---------------------------------------------------------------------------
# 单个 λ 的反演与诊断 (对应 .m 循环体)
# ---------------------------------------------------------------------------

def invert_at_lambda(
    snap: Dict[str, Any],
    lam: float,
    *,
    Con: Sequence[int] = (0, 0, 0),
    slip_max: float = 10.0,
    max_nfev: int = 100,
    mu: float = 30e9,
    verbose: bool = True,
) -> Dict[str, Any]:
    slip_model = snap["slip_model"]
    h1 = snap["h1"]
    add_col = _add_col_of(snap["ramp_choice"])
    n_classes = int(snap["class_map"].max()) if snap["class_map"].size else 1
    total_ramp_cols = add_col * max(n_classes, 1)

    Greens = np.vstack([
        snap["G_last"],
        snap["H"] * (lam / h1),
        snap["Wb"], snap["Wl"], snap["Wr"],
    ])
    b = snap["bdata_sm"]

    fault_id = slip_model[:, 0].astype(int)
    nflt = int(slip_model[:, 0].max())
    tSm = np.zeros(nflt + 1, dtype=int)
    for i in range(1, nflt + 1):
        tSm[i] = int(np.sum(fault_id == i))
    lb, ub = bounds_new(nflt, 2, tSm, slip_max, total_ramp_cols, Con)

    u, msg = _solve_bounded_lsq(Greens, b, lb, ub, max_nfev, verbose=verbose)

    # --- 诊断量: 公式与 make_fault_from_insar1 / make_rms.m 一致 ---
    Bdata = snap["Bdata"].ravel()
    bd_last = snap["bd_last"].ravel()
    GrF, G_last, H = snap["GrF"], snap["G_last"], snap["H"]

    resid_raw = GrF @ u - Bdata
    rms0 = float(np.sum(Bdata * Bdata))
    rms1 = float(np.sum(resid_raw * resid_raw))
    redu_perc = 100.0 * (rms0 - rms1) / rms0 if rms0 else 0.0
    rms_val = float(np.sqrt(np.mean(resid_raw * resid_raw)))

    RMS_misfit = float(np.sum((G_last @ u - bd_last) ** 2))
    rough_matrix = H @ u
    roughness = float(np.sqrt(np.mean(rough_matrix * rough_matrix))) \
        if rough_matrix.size else 0.0

    Npatch = int(tSm.sum())
    slip_lam = slip_model.copy()
    slip_lam[:, 11] = u[:Npatch]
    slip_lam[:, 12] = u[Npatch:2 * Npatch]

    strike_u, strike_d = slip_lam[:, 11], slip_lam[:, 12]
    max_slip = float(np.max(np.sqrt(strike_u ** 2 + strike_d ** 2)))
    Apatch = slip_lam[:, 6] * slip_lam[:, 7]
    D = np.sqrt(strike_u ** 2 + strike_d ** 2)
    M0 = float(np.sum(mu * D * Apatch))
    Mw = 2.0 / 3.0 * (np.log10(M0) - 9.1) if M0 > 0 else float("nan")

    if verbose:
        print(
            "lambda=%-8g rms(dat., res.)=%.6e -> %.6e (%.4f%%)  RMS=%.6f  "
            "roughness=%.6f  max_slip=%.4f  Mw=%.4f"
            % (lam, rms0, rms1, redu_perc, rms_val, roughness, max_slip, Mw),
            flush=True,
        )

    return dict(
        lam=float(lam), u=u, rms=rms_val, rms0=rms0, rms1=rms1,
        redu_perc=redu_perc, RMS_misfit=RMS_misfit, roughness=roughness,
        max_slip=max_slip, M0=M0, Mw=Mw, slip_model=slip_lam, solver_msg=msg,
    )


def scan_lambdas(
    snap: Dict[str, Any],
    lambdas: Sequence[float],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for lam in lambdas:
        print("=== lambda = %g ===" % lam, flush=True)
        results.append(invert_at_lambda(snap, lam, **kwargs))
    return results


# ---------------------------------------------------------------------------
# 曲线与表格
# ---------------------------------------------------------------------------

def save_scan_csv(results: Sequence[Dict[str, Any]], out_csv: str) -> None:
    fields = ("lam", "rms", "redu_perc", "roughness", "RMS_misfit",
              "Mw", "M0", "max_slip")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            w.writerow([r[k] for k in fields])
    print("scan table -> %s" % out_csv, flush=True)


def plot_curves(results: Sequence[Dict[str, Any]], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    lams = np.array([r["lam"] for r in results], dtype=np.float64)
    rms = np.array([r["rms"] for r in results], dtype=np.float64)
    redu = np.array([r["redu_perc"] for r in results], dtype=np.float64)
    rough = np.array([r["roughness"] for r in results], dtype=np.float64)
    x = np.log10(lams)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 8.0), sharex=True)
    ax1.plot(x, rms, "-o", linewidth=1.5)
    ax1.set_ylabel("RMS residual (data units)")
    ax1.grid(True)
    ax2.plot(x, redu, "-o", linewidth=1.5)
    ax2.set_ylabel("redu_perc (%)")
    ax2.set_xlabel("Smoothness (log10)")
    ax2.grid(True)
    if np.any(np.abs(lams - 1.0) < 1e-9):
        for ax in (ax1, ax2):
            ax.axvline(0.0, color="0.6", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    out1 = os.path.join(out_dir, "rms_vs_lambda.png")
    fig.savefig(out1, dpi=200, facecolor="w")
    plt.close(fig)
    print("curve -> %s" % out1, flush=True)

    fig2, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(rough, rms, "-o", linewidth=1.5)
    for r in results:
        ax.annotate("%g" % r["lam"], (r["roughness"], r["rms"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Model roughness")
    ax.set_ylabel("RMS residual (data units)")
    ax.set_title("L-curve")
    ax.grid(True)
    fig2.tight_layout()
    out2 = os.path.join(out_dir, "lcurve.png")
    fig2.savefig(out2, dpi=200, facecolor="w")
    plt.close(fig2)
    print("curve -> %s" % out2, flush=True)


# ---------------------------------------------------------------------------
# 逐轨道差异图
# ---------------------------------------------------------------------------

def _fault_segments_km(
    fault_path: str,
    ref_lon: float,
    lonc: float,
    latc: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """断层迹线 -> [(xs_km, ys_km), ...], 以 (lonc, latc) 为原点 (与 .m 绘图一致)."""
    from load_fault_one_plane import _ll2xy

    seg = np.loadtxt(fault_path, dtype=np.float64)
    if seg.ndim == 1:
        seg = seg.reshape(1, -1)
    if seg.shape[1] < 4:
        raise ValueError("fault 文件每行需至少 4 列: %s" % fault_path)
    xo, yo = _ll2xy(np.array([lonc]), np.array([latc]), ref_lon)
    lonf = np.concatenate([seg[:, 0], seg[:, 2]])
    latf = np.concatenate([seg[:, 1], seg[:, 3]])
    nseg = int(lonf.size // 2)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for ii in range(nseg):
        slon = np.array([lonf[ii], lonf[ii + nseg]])
        slat = np.array([latf[ii], latf[ii + nseg]])
        xx, yy = _ll2xy(slon, slat, ref_lon)
        xs = (np.asarray(xx).ravel() - np.ravel(xo)[0]) / 1000.0
        ys = (np.asarray(yy).ravel() - np.ravel(yo)[0]) / 1000.0
        out.append((xs, ys))
    return out


def resolve_track_files(
    config_dir: str,
    data_list: str,
    default_region: Tuple[float, float, float, float],
    iint: int,
    tracks_override: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """按 data_list 顺序解析各道采样文件; 目录缺失时按轨道名在 input_data 下搜回.

    返回 [{name, path, dtype}, ...]; 候选文件依次为 los_samp<iint>.mat, los_samp.mat.
    """
    from inversion_example_workflow import read_data_list

    data_list_abs = data_list if os.path.isabs(data_list) \
        else os.path.join(config_dir, data_list)
    tracks, _, _, dtypes, _, _ = read_data_list(
        data_list_abs, default_region=default_region,
    )
    if tracks_override is not None:
        if len(tracks_override) != len(dtypes):
            raise ValueError(
                "--tracks-dir 给了 %d 个目录, 与 data_list 的 %d 道不一致"
                % (len(tracks_override), len(dtypes))
            )
        tracks = [os.path.normpath(p) for p in tracks_override]

    candidates = ("los_samp%d.mat" % int(iint), "los_samp.mat")
    out: List[Dict[str, Any]] = []
    for tr, dtp in zip(tracks, dtypes):
        chosen: Optional[str] = None
        for cand in candidates:
            p = os.path.join(tr, cand)
            if os.path.isfile(p):
                chosen = p
                break
        if chosen is None:
            # 目录不在 (如 datav0.list 指向已改名的 data_v2): 按轨道名搜索
            name = os.path.basename(os.path.normpath(tr))
            search_root = os.path.join(config_dir, "input_data")
            if not os.path.isdir(search_root):
                search_root = config_dir
            for dirpath, dirnames, filenames in os.walk(search_root):
                dirnames[:] = [dd for dd in dirnames if dd != "__pycache__"]
                if os.path.basename(dirpath) != name:
                    continue
                for cand in candidates:
                    if cand in filenames:
                        chosen = os.path.join(dirpath, cand)
                        break
                if chosen:
                    break
        if chosen is None:
            raise FileNotFoundError(
                "轨道 %s 下找不到 %s; 请重跑 Step 6 (resamp) 或用 --tracks-dir 指定"
                % (tr, " / ".join(candidates))
            )
        out.append(dict(name=os.path.basename(os.path.normpath(tr)),
                        path=chosen, dtype=dtp))
    return out


def _load_track_sids(track_files: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
    sids = []
    for tf in track_files:
        d = loadmat(tf["path"])
        if "sampled_insar_data" not in d:
            raise KeyError("%s 中无 sampled_insar_data" % tf["path"])
        sids.append(np.asarray(d["sampled_insar_data"], dtype=np.float64))
    return sids


def _check_track_rows(
    track_files: Sequence[Dict[str, Any]], n_expected: int, iint: int
) -> Optional[str]:
    """各道行数之和须等于快照 GrF 行数; 不一致时返回说明文字, 一致返回 None."""
    try:
        sids = _load_track_sids(track_files)
    except KeyError as e:
        return str(e)
    counts = [(tf["name"], int(sid.shape[0])) for tf, sid in zip(track_files, sids)]
    total = int(sum(sid.shape[0] for sid in sids))
    if total != n_expected:
        return (
            "各道采样点数 %s 合计 %d, 与快照观测行数 %d 不一致 "
            "(快照来自另一套 los_samp 数据); 请重跑 Step 6 生成 los_samp%d.mat, "
            "或改用与本地数据匹配的快照 (--snapshot py_inversion_iint0.mat --iint 0)"
            % (counts, total, n_expected, iint)
        )
    return None


def plot_track_misfits(
    result: Dict[str, Any],
    snap: Dict[str, Any],
    track_files: Sequence[Dict[str, Any]],
    *,
    out_dir: str,
    iint: int,
    misfit_range: float,
    defo_max: float,
    ref_lon: float,
    lonc: float,
    latc: float,
    fault_abs: str,
) -> None:
    """指定 λ 的逐轨道 观测/模型/残差 三联图 + 残差汇总图 (均叠断层迹线)."""
    from plot_insar_model_resampled import plot_insar_model_resampled

    lam = result["lam"]
    lam_dir = os.path.join(out_dir, "lam_%.3f" % lam)
    os.makedirs(lam_dir, exist_ok=True)

    GrF, u = snap["GrF"], result["u"]
    sids = _load_track_sids(track_files)
    blocks: List[Tuple[Dict[str, Any], np.ndarray, int, int]] = []
    i0 = 0
    for tf, sid in zip(track_files, sids):
        n = int(sid.shape[0])
        blocks.append((tf, sid, i0, i0 + n))
        i0 += n

    fault_ok = bool(fault_abs) and os.path.isfile(fault_abs)
    segs = _fault_segments_km(fault_abs, ref_lon, lonc, latc) if fault_ok else []

    # --- 每道三联图 (与 .m 末段 plot_insar_model_resampled 一致) ---
    for tf, sid, i0, i1 in blocks:
        model_i = (GrF[i0:i1, :] @ u).ravel()
        resid_i = sid[:, 2] - model_i
        out_png = os.path.join(lam_dir, "track_%s_misfit.png" % tf["name"])
        plot_insar_model_resampled(
            tf["path"],
            model_i,
            iter_step=iint,
            fault=fault_abs if fault_ok else None,
            misfit_range=misfit_range,
            defo_max=defo_max,
            ref_lon=ref_lon,
            lonc=lonc,
            latc=latc,
            out_figure=out_png,
            save_los_model_mat=False,
            show=False,
        )
        print(
            "  track %s: n=%d  RMS=%.6f  -> %s"
            % (tf["name"], i1 - i0,
               float(np.sqrt(np.mean(resid_i * resid_i))), out_png),
            flush=True,
        )

    # --- 残差汇总图: 每道一个子图, 统一色标, 叠断层 ---
    import matplotlib.pyplot as plt

    n_tr = len(blocks)
    ncol = 2 if n_tr > 1 else 1
    nrow = int(np.ceil(n_tr / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 5.6 * nrow),
                             squeeze=False)
    all_resid: List[np.ndarray] = []
    for ax, (tf, sid, i0, i1) in zip(axes.ravel(), blocks):
        xin, yin = sid[:, 0] / 1000.0, sid[:, 1] / 1000.0
        resid = sid[:, 2] - (GrF[i0:i1, :] @ u).ravel()
        all_resid.append(resid)
        sc = ax.scatter(xin, yin, c=resid, cmap="jet", s=4)
        sc.set_clim(-misfit_range, misfit_range)
        for xs, ys in segs:
            ax.plot(xs, ys, c="k", linewidth=1.5)
        ax.set_title(
            "%s  RMS=%.4g  (n=%d)"
            % (tf["name"], float(np.sqrt(np.mean(resid ** 2))), i1 - i0),
            fontsize=11,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, shrink=0.75)
    for ax in axes.ravel()[n_tr:]:
        ax.set_visible(False)
    vmax = float(np.max(np.abs(np.concatenate(all_resid)))) if all_resid else 1.0
    fig.suptitle(
        "Per-track residual, lambda=%g  (clim=±%g; data range ±%.3g)"
        % (lam, misfit_range, vmax),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = os.path.join(lam_dir, "tracks_residual.png")
    fig.savefig(out_png, dpi=200, facecolor="w")
    plt.close(fig)
    print("tracks residual -> %s" % out_png, flush=True)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def _parse_floats(text: str) -> List[float]:
    return [float(tok) for tok in re.split(r"[,\s]+", text.strip()) if tok]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="make_rms.m 的 Python 移植: 对 Step 7 反演快照做平滑系数 λ 扫描, "
                    "输出 RMS-λ 曲线、各 λ 滑移模型与逐轨道差异图.")
    ap.add_argument("--snapshot", default="py_inversion_iint1.mat",
                    help="反演快照 .mat (默认 py_inversion_iint1.mat, 即 Step 7)")
    ap.add_argument("--config-dir", default=_HERE,
                    help="configfile.txt / configpara.txt 所在目录 (默认脚本目录)")
    ap.add_argument("--lambdas", default="0.1,0.5,0.8,1.0,1.2,1.5",
                    help="逗号分隔的平滑系数序列")
    ap.add_argument("--con", default="0,0,0", help="滑移符号约束 strike,dip,normal")
    ap.add_argument("--slip-max", type=float, default=10.0, help="滑移上界 (m)")
    ap.add_argument("--max-nfev", type=int, default=None,
                    help="求解器最大迭代 (默认取 configpara 的 max_nfev)")
    ap.add_argument("--mu", type=float, default=30e9, help="剪切模量 (Pa)")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "make_rms_out"))
    ap.add_argument("--misfit-lambdas", default="1.0",
                    help="需要画逐轨道差异图的 λ (逗号分隔; 若不在 --lambdas 中则现场补算)")
    ap.add_argument("--misfit-range", type=float, default=1.0,
                    help="残差图色标 ±misfit_range (数据单位)")
    ap.add_argument("--defo-max", type=float, default=3.0,
                    help="观测/模型图色标 ±defo_max (数据单位)")
    ap.add_argument("--iint", type=int, default=None,
                    help="采样文件序号 (默认从快照文件名 iint\\d+ 推断)")
    ap.add_argument("--tracks-dir", default=None,
                    help="逗号分隔的轨道目录列表, 覆盖 data_list 中的路径 (顺序须一致)")
    ap.add_argument("--no-slip-figs", action="store_true",
                    help="不画各 λ 的 3D 滑移模型")
    ap.add_argument("--no-misfit", action="store_true", help="不画逐轨道差异图")
    ap.add_argument("--show", action="store_true", help="弹窗显示 (默认只存文件)")
    args = ap.parse_args(argv)

    if not args.show:
        import matplotlib
        try:
            matplotlib.use("Agg", force=True)
        except TypeError:
            matplotlib.use("Agg")

    from inversion_example_workflow import WorkflowConfig, load_workflow_config

    # --- 配置 (参考点 / 断层文件 / data_list) ---
    try:
        _root, cfg = load_workflow_config(args.config_dir)
    except (FileNotFoundError, ValueError):
        cfg = WorkflowConfig()
        _root = args.config_dir
    ref_lon, lonc, latc = cfg.ref_lon, cfg.lonc, cfg.latc
    max_nfev = args.max_nfev if args.max_nfev is not None else cfg.max_nfev
    fault_abs = cfg.fault_file
    if fault_abs and not os.path.isabs(fault_abs):
        fault_abs = os.path.normpath(os.path.join(_root, fault_abs))

    # --- 快照 (路径相对脚本目录解析) ---
    snap_path = args.snapshot
    if not os.path.isabs(snap_path):
        snap_path = os.path.join(_HERE, snap_path)
    print("加载快照: %s" % snap_path, flush=True)
    snap = load_snapshot(snap_path)
    print(
        "ramp_choice=%s  obs=%d  patches=%d  cols=%d"
        % (snap["ramp_choice"], snap["G_last"].shape[0],
           snap["slip_model"].shape[0], snap["G_last"].shape[1]),
        flush=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # --- λ 扫描 ---
    lambdas = _parse_floats(args.lambdas)
    con = tuple(int(x) for x in _parse_floats(args.con)[:3])
    results = scan_lambdas(
        snap, lambdas, Con=con, slip_max=args.slip_max,
        max_nfev=max_nfev, mu=args.mu,
    )

    # 参考检查: λ=1.0 对应工作流 Step 7 的 smoothness 默认值; 因快照生成时的
    # 求解器状态/容差可能不同, 病态问题下允许有小差异, 仅数量级参考.
    u_ref = np.asarray(snap.get("u", np.zeros(0)), dtype=np.float64).ravel()
    lam_ref = next((r for r in results if abs(r["lam"] - 1.0) < 1e-9), None)
    if lam_ref is not None and u_ref.size == lam_ref["u"].size:
        print("参考: λ=1.0 重算解与快照 u 的最大偏差 %.3e (病态问题, 仅数量级参考)"
              % float(np.max(np.abs(lam_ref["u"] - u_ref))), flush=True)

    save_scan_csv(results, os.path.join(args.out_dir, "scan_table.csv"))
    plot_curves(results, args.out_dir)

    # --- 各 λ 的 3D 滑移模型 ---
    if not args.no_slip_figs:
        from show_slip_model import show_slip_model
        for r in results:
            out_png = os.path.join(args.out_dir, "slip_lam_%.3f.png" % r["lam"])
            show_slip_model(
                r["slip_model"],
                ref_lon=ref_lon, lonc=lonc, latc=latc,
                fault=fault_abs if fault_abs and os.path.isfile(fault_abs) else None,
                out_path=out_png, show=False, block=False,
                title="slip model (lambda=%g, RMS=%.4f)" % (r["lam"], r["rms"]),
            )
            print("slip model -> %s" % out_png, flush=True)

    # --- 逐轨道差异图 ---
    if not args.no_misfit:
        iint = args.iint
        if iint is None:
            m = re.search(r"iint(\d+)", os.path.basename(snap_path))
            iint = int(m.group(1)) if m else 1
        override: Optional[List[str]] = None
        if args.tracks_dir:
            override = [p for p in args.tracks_dir.split(",") if p.strip()]
        row_err: Optional[str] = None
        track_files: List[Dict[str, Any]] = []
        try:
            track_files = resolve_track_files(
                args.config_dir, cfg.data_list, cfg.default_region, iint,
                tracks_override=override,
            )
            row_err = _check_track_rows(track_files, snap["GrF"].shape[0], iint)
        except (FileNotFoundError, ValueError, KeyError) as e:
            row_err = str(e)
        if row_err is not None:
            print("!! 跳过逐轨道差异图: %s" % row_err, flush=True)
        else:
            for lam_want in _parse_floats(args.misfit_lambdas):
                r = next((x for x in results
                          if abs(x["lam"] - lam_want) < 1e-9), None)
                if r is None:
                    print("misfit λ=%g 不在扫描序列中, 现场补算..." % lam_want,
                          flush=True)
                    r = invert_at_lambda(
                        snap, lam_want, Con=con, slip_max=args.slip_max,
                        max_nfev=max_nfev, mu=args.mu,
                    )
                plot_track_misfits(
                    r, snap, track_files,
                    out_dir=args.out_dir, iint=iint,
                    misfit_range=args.misfit_range, defo_max=args.defo_max,
                    ref_lon=ref_lon, lonc=lonc, latc=latc, fault_abs=fault_abs,
                )

    print("make_rms 完成, 输出目录: %s" % os.path.normpath(args.out_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
