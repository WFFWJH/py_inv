r"""用 Step 5 反演对四段断层倾角做网格搜索 (参照 ``inversion_example_workflow.py``).

原理: 倾角组合只改变 Step 4 断层构建 (``load_fault_one_plane`` 的 ``dip`` 参数),
数据侧不变 (Step 3 已有的各道 ``los_samp<iter_step>.mat``)。对每个组合::

    Step 0: step_load_config (一次)
    Step 4: step_load_fault      ← dip 变化点
    Step 5: make_fault_from_insar1(iter_step=0) → RMS / redu% / roughness / Mw

搜索模式
--------
``each``  逐段扫描: 其余段固定为当前基线, 只扫第 i 段的范围; 共 Σn_i 个组合。
          配合 ``--rounds N`` 做坐标下降: 每轮结束后把各段更新为当轮最优, 再扫下一轮。
``grid``  四段范围的笛卡尔积全网格 (Πn_i 个组合; 大网格很慢, 启动时会提示预计耗时)。

用法::

    python dip_grid_search.py                                    # 默认每段 74,78,82,86,90
    python dip_grid_search.py --seg1 70:90:5 --seg2 82 --seg3 82 --seg4 82
    python dip_grid_search.py --mode grid --seg1 75,80,85 --seg2 78,82 --seg3 85,90 --seg4 80
    python dip_grid_search.py --rounds 3 --backend numba         # 坐标下降 + numba 加速

倾角范围写法: ``82`` 单值 / ``75,80,85`` 枚举 / ``70:90:5`` 等差 (含端点)。

输出 (默认 ``dip_search_out/``):
    ``dip_scan_table.csv``    每个组合的诊断量表 (增量写出, 可中断续看)
    ``rms_per_segment.png``   各段 RMS–倾角曲线 (each: 最后一轮; grid: 其他维取最小的边际曲线)
    ``best_slip_model.png``   最优组合的 3D 滑移模型
    ``py_dip_search_best.mat`` 最优组合的完整快照 (与工作流 .mat 同结构)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bounds_new import bounds_new  # noqa: F401  (经 make_fault_from_insar1 间接使用, 保持一致性)
from inversion_example_workflow import (
    InversionWorkflowState,
    _check_los_samp_files,
    _save_inversion_mat,
    step_load_config,
    step_load_fault,
)
from make_fault_from_insar1 import make_fault_from_insar1
from show_slip_model import show_slip_model


# ---------------------------------------------------------------------------
# 倾角范围解析: "82" / "75,80,85" / "70:90:5" (等差, 含端点)
# ---------------------------------------------------------------------------

def parse_dip_spec(text: str, seg_idx: int) -> List[float]:
    text = text.strip()
    if not text:
        raise ValueError("--seg%d 为空" % (seg_idx + 1))
    m = re.fullmatch(r"(-?[\d.]+)\s*:\s*(-?[\d.]+)\s*:\s*(-?[\d.]+)", text)
    if m:
        lo, hi, step = (float(g) for g in m.groups())
        if step <= 0 or hi < lo:
            raise ValueError("--seg%d 等差范围无效: %s" % (seg_idx + 1, text))
        vals = np.arange(lo, hi + 0.5 * step, step)
        return [float(round(v, 6)) for v in vals]
    return [float(tok) for tok in re.split(r"[,\s]+", text) if tok]


# ---------------------------------------------------------------------------
# datav0.list 可能指向已改名/删除的目录 (如 data_v2): 按轨道名在 input_data 下搜回
# ---------------------------------------------------------------------------

def fix_track_paths(state: InversionWorkflowState, config_dir: str) -> None:
    iint = int(state.config.iter_step)
    need = "los_samp%d.mat" % iint
    fixed: List[str] = []
    for tr in state.tracks:
        if os.path.isfile(os.path.join(tr, need)):
            fixed.append(tr)
            continue
        name = os.path.basename(os.path.normpath(tr))
        search_root = os.path.join(config_dir, "input_data")
        if not os.path.isdir(search_root):
            search_root = config_dir
        found: Optional[str] = None
        for dirpath, dirnames, filenames in os.walk(search_root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            if os.path.basename(dirpath) == name and need in filenames:
                found = dirpath
                break
        if found is None:
            fixed.append(tr)  # 保持原样, 让后续 _check_los_samp_files 给出明确报错
        else:
            print("  [track] %s 不可用, 改用 %s" % (tr, found), flush=True)
            fixed.append(found)
    state.tracks = fixed


# ---------------------------------------------------------------------------
# 单个倾角组合: Step 4 (几何) + Step 5 (反演)
# ---------------------------------------------------------------------------

def run_combo(
    state: InversionWorkflowState,
    dips: Sequence[float],
    *,
    backend: str,
    max_nfev: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    cfg = state.config
    state.dangles = [float(x) for x in dips]
    step_load_fault(state)  # Step 4: 重建断层几何 (倾角变化点)

    slip, rms_val, rough, ret, extras = make_fault_from_insar1(
        state.slip_vs, None, int(cfg.iter_step), state.tracks,
        paths_type=state.dtypes,
        ramp_choice=cfg.ramp_choice,
        segment_smooth_file=state.seg_file_abs,
        intersect_smooth_file=None,
        fault_file=state.fault_abs,
        ref_lon=cfg.ref_lon, lonc=cfg.lonc, latc=cfg.latc,
        Con=cfg.con,
        model_type=cfg.model_type,
        backend=backend,
        max_nfev=max_nfev,
        verbose=verbose,
        plot_resampled_fits=False,  # 搜索模式不逐道出图 (避免每组参数写 4 张图)
    )

    u = np.asarray(extras["u"], dtype=np.float64).ravel()
    Npatch = slip.shape[0]
    strike_u, strike_d = u[:Npatch], u[Npatch:2 * Npatch]
    max_slip = float(np.max(np.sqrt(strike_u ** 2 + strike_d ** 2)))

    return dict(
        dips=[float(x) for x in dips],
        rms=float(rms_val),
        redu_perc=float(ret[0]),
        exitflag=float(ret[1]),
        Mw=float(ret[3]),
        rough=float(rough),
        max_slip=max_slip,
        slip=slip,
        rms_misfit_chi2=float(np.sum((extras["G_last"] @ u - extras["bd_last"]) ** 2)),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

_CSV_FIELDS = ("mode", "round", "seg_varied", "d1", "d2", "d3", "d4",
               "rms", "redu_perc", "roughness", "Mw", "max_slip", "exitflag")


class CsvWriter:
    """增量写 CSV, 长搜索中断不丢数据."""

    def __init__(self, path: str) -> None:
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(_CSV_FIELDS)
        self._f.flush()
        print("scan table -> %s" % path, flush=True)

    def add(self, rec: Dict[str, Any]) -> None:
        self._w.writerow([
            rec.get("mode", ""), rec.get("round", ""), rec.get("seg_varied", ""),
            *rec["dips"], rec["rms"], rec["redu_perc"], rec["rough"],
            rec["Mw"], rec["max_slip"], rec["exitflag"],
        ])
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def plot_scan(results: Sequence[Dict[str, Any]], nseg: int, out_png: str,
              mode: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), squeeze=False)
    for i in range(nseg):
        ax = axes[i // 2][i % 2]
        if mode == "grid":
            # 边际曲线: 固定第 i 段倾角, 其余维取最小 RMS
            by_dip: Dict[float, float] = {}
            for r in results:
                d = r["dips"][i]
                by_dip[d] = min(by_dip.get(d, float("inf")), r["rms"])
            xs = sorted(by_dip)
            ys = [by_dip[x] for x in xs]
            label = "min RMS over other segments"
        else:
            # each: 取最后一轮该段的扫描点
            last_round = max(r["round"] for r in results)
            recs = sorted(
                (r for r in results
                 if r.get("seg_varied") == i + 1 and r["round"] == last_round),
                key=lambda r: r["dips"][i],
            )
            xs = [r["dips"][i] for r in recs]
            ys = [r["rms"] for r in recs]
            label = "RMS (others at current best)"
        if xs:
            ax.plot(xs, ys, "-o", linewidth=1.5)
            k = int(np.argmin(ys))
            ax.scatter([xs[k]], [ys[k]], marker="*", s=220, color="crimson",
                       zorder=5, label="best %.0f°" % xs[k])
        ax.set_title("Segment %d" % (i + 1))
        ax.set_xlabel("dip (deg)")
        ax.set_ylabel("RMS (data units)")
        ax.grid(True, alpha=0.4)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)
    for ax in axes.ravel()[nseg:]:
        ax.set_visible(False)
    fig.suptitle("Dip grid search via Step-5 inversion (%s mode)" % mode, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200, facecolor="w")
    plt.close(fig)
    print("plot -> %s" % out_png, flush=True)


def report_best(rec: Dict[str, Any], elapsed_per_combo: float, n_total: int) -> None:
    print("\n===== 最优组合 =====", flush=True)
    print("dips = %s" % (rec["dips"],), flush=True)
    print(
        "RMS=%.6f  redu=%.4f%%  roughness=%.6f  max_slip=%.4f  Mw=%.4f  exitflag=%d"
        % (rec["rms"], rec["redu_perc"], rec["rough"], rec["max_slip"],
           rec["Mw"], rec["exitflag"]),
        flush=True,
    )
    print("平均每组耗时 %.1f s × 共 %d 组" % (elapsed_per_combo, n_total), flush=True)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="四段断层倾角网格搜索 (Step 5 反演评价; 只改 Step 4 几何构建)")
    ap.add_argument("--config-dir", default=_HERE)
    ap.add_argument("--mode", choices=("each", "grid"), default="each",
                    help="each=逐段扫描(默认, 省 时); grid=四段全网格")
    ap.add_argument("--seg1", default="74,78,82,86,90", help="第 1 段倾角范围")
    ap.add_argument("--seg2", default="74,78,82,86,90", help="第 2 段倾角范围")
    ap.add_argument("--seg3", default="74,78,82,86,90", help="第 3 段倾角范围")
    ap.add_argument("--seg4", default="74,78,82,86,90", help="第 4 段倾角范围")
    ap.add_argument("--baseline", default=None,
                    help="each 模式的基线倾角 (默认取 configpara 的 ##dip)")
    ap.add_argument("--rounds", type=int, default=1,
                    help="each 模式的坐标下降轮数 (每轮后各段更新为当轮最优)")
    ap.add_argument("--backend", default=None,
                    help="okada 后端 (默认取 configpara 的 okada_backend; 可填 numba 加速)")
    ap.add_argument("--max-nfev", type=int, default=None,
                    help="求解器最大迭代 (默认取 configpara)")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "dip_search_out"))
    ap.add_argument("--show", action="store_true", help="弹窗显示 (默认只存文件)")
    args = ap.parse_args(argv)

    if not args.show:
        import matplotlib
        try:
            matplotlib.use("Agg", force=True)
        except TypeError:
            matplotlib.use("Agg")

    # --- Step 0: 读配置 (基线倾角来自 configpara ##dip 或 --baseline) ---
    baseline_dips = ([float(x) for x in _parse_list(args.baseline)]
                     if args.baseline else None)
    state = InversionWorkflowState()
    step_load_config(state, args.config_dir, dip_per_segment=baseline_dips)
    cfg = state.config
    nseg = state.nseg
    if nseg > 4:
        raise ValueError("当前脚本按 4 段断层设计 (--seg1..--seg4), 实际 %d 段" % nseg)
    baseline = list(state.dangles)
    seg_lists = [parse_dip_spec(s, i) for i, s in
                 enumerate((args.seg1, args.seg2, args.seg3, args.seg4)[:nseg])]

    # --- 数据可用性: 修正 datav0.list 可能失效的轨道路径, 并快速校验 ---
    fix_track_paths(state, args.config_dir)
    _check_los_samp_files(state, int(cfg.iter_step))

    backend = args.backend or cfg.okada_backend
    max_nfev = args.max_nfev if args.max_nfev is not None else cfg.max_nfev

    n_total = (sum(len(v) for v in seg_lists) if args.mode == "each"
               else int(np.prod([len(v) for v in seg_lists])))
    print("mode=%s  baseline=%s  seg ranges=%s  共 %d 组  backend=%s"
          % (args.mode, baseline, seg_lists, n_total, backend), flush=True)
    if args.mode == "grid" and n_total > 60:
        print("!! 全网格组合较多, 如耗时过长可改用 --mode each 或缩小范围", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csvw = CsvWriter(os.path.join(args.out_dir, "dip_scan_table.csv"))

    results: List[Dict[str, Any]] = []
    measured: Dict[Tuple[float, ...], Dict[str, Any]] = {}
    best: Optional[Dict[str, Any]] = None
    best_extras: Optional[Dict[str, Any]] = None
    t_first: Optional[float] = None

    def do_combo(dips: Sequence[float], **meta: Any) -> Dict[str, Any]:
        nonlocal best, t_first, best_extras
        key = tuple(round(float(d), 6) for d in dips)
        if key in measured:
            return measured[key]
        t0 = time.perf_counter()
        rec = run_combo(state, dips, backend=backend, max_nfev=max_nfev)
        dt = time.perf_counter() - t0
        if t_first is None:
            t_first = dt
            print("  (预计剩余约 %.0f 分钟)"
                  % (dt * (n_total - len(measured) - 1) / 60.0), flush=True)
        rec.update(meta)
        rec["dips"] = [float(d) for d in dips]
        results.append(rec)
        measured[key] = rec
        csvw.add(rec)
        print("[%s] dips=%s  RMS=%.6f  redu=%.3f%%  rough=%.4f  Mw=%.4f  (%.1fs)"
              % (meta.get("tag", ""), rec["dips"], rec["rms"], rec["redu_perc"],
                 rec["rough"], rec["Mw"], dt), flush=True)
        if best is None or rec["rms"] < best["rms"]:
            best = rec
            # extras 含 G_last/GrF/H 等大矩阵 (~百 MB/组), 只保留当前最优的
            best_extras = rec.pop("extras")
        else:
            rec.pop("extras")
        return rec

    try:
        if args.mode == "each":
            cur = baseline[:]
            for rnd in range(1, max(1, args.rounds) + 1):
                print("\n--- round %d (baseline=%s) ---" % (rnd, cur), flush=True)
                for i in range(nseg):
                    for d in seg_lists[i]:
                        dips = cur[:]
                        dips[i] = d
                        do_combo(dips, mode="each", round=rnd, seg_varied=i + 1,
                                 tag="r%d.s%d" % (rnd, i + 1))
                    # 该段更新为当轮最优 (在其扫描点中)
                    seg_recs = [r for r in results
                                if r.get("seg_varied") == i + 1 and r["round"] == rnd]
                    if seg_recs:
                        cur[i] = min(seg_recs, key=lambda r: r["rms"])["dips"][i]
                print("round %d 结束: 当前最优 = %s" % (rnd, cur), flush=True)
        else:
            for dips in itertools.product(*seg_lists):
                do_combo(dips, mode="grid", round=1, seg_varied=0, tag="grid")
    finally:
        csvw.close()

    assert best is not None and best_extras is not None
    report_best(best, t_first or 0.0, len(results))

    # --- 最优组合的产出物 ---
    out_mat = os.path.join(args.out_dir, "py_dip_search_best.mat")
    _save_inversion_mat(
        out_mat, best["slip"], best["rms"], best["rough"],
        np.asarray([best["redu_perc"], best["exitflag"], best["rms"],
                    best["Mw"], 0.0], dtype=np.float64),
        best_extras, ramp_choice=cfg.ramp_choice,
    )
    out_png = os.path.join(args.out_dir, "best_slip_model.png")
    show_slip_model(
        best["slip"],
        ref_lon=cfg.ref_lon, lonc=cfg.lonc, latc=cfg.latc,
        fault=state.fault_abs if os.path.isfile(state.fault_abs) else None,
        out_path=out_png, show=False, block=False,
        title="best dips %s (RMS=%.4f)" % (best["dips"], best["rms"]),
    )
    print("best slip model -> %s" % out_png, flush=True)
    plot_scan(results, nseg, os.path.join(args.out_dir, "rms_per_segment.png"),
              args.mode)

    print("\ndip 搜索完成, 输出目录: %s" % os.path.normpath(args.out_dir), flush=True)
    return 0


def _parse_list(text: str) -> List[float]:
    return [float(tok) for tok in re.split(r"[,\s]+", text.strip()) if tok]


if __name__ == "__main__":
    raise SystemExit(main())
