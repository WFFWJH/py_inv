r"""与 ``plot_insar_model_resampled.m`` 一致: 读取每道 ``los_samp*.mat``, 绘制观测 / 模型 / 残差.

- 观测: ``sampled_insar_data(:,3)``
- 模型: 参数 ``los_model`` (与 ``G_raw_ramp{i}*u`` 同维)
- 残差: 观测 − 模型

可选写出 ``los_model.mat`` / ``los_model_layered.mat`` (``sampled_model`` 列与 .m 相同).
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Union

import numpy as np
from scipy.io import loadmat, savemat

from load_fault_one_plane import _ll2xy


def plot_insar_model_resampled(
    sampled_data_file: Union[str, os.PathLike],
    los_model: np.ndarray,
    *,
    iter_step: int = 0,
    fault: Optional[Union[str, os.PathLike]] = None,
    model_type: str = "okada",
    misfit_range: float = 100.0,
    defo_max: float = 300.0,
    ref_lon: float = 95.0,
    lonc: float = 95.33,
    latc: float = 19.61,
    axis_range: Optional[Sequence[float]] = None,
    out_figure: Optional[Union[str, os.PathLike]] = None,
    save_los_model_mat: bool = True,
    show: bool = False,
) -> Any:
    """绘制每道重采样点的观测、合成模型与残差 (三子图), 可选保存 PNG 与 ``los_model*.mat``.

    Parameters
    ----------
    sampled_data_file
        含 ``sampled_insar_data`` 的 ``.mat`` (与 ``resamp_insar_data`` / MATLAB 一致).
    los_model
        与观测等长的模型列向量 (未加权; 与 ``G_raw_ramp{i} @ u`` 一致).
    fault
        断层 4 列文件 (``lon1 lat1 lon2 lat2``/行), 同 ``load_fault_one_plane``; 在 km 平面上叠画迹线.
    misfit_range
        残差子图色标 :math:`\\pm` ``misfit_range`` (与 .m 中 ``res_max``).
    defo_max
        观测/模型子图色标为 ``[-defo_max, defo_max]``.
    out_figure
        若给定则 ``savefig``; 未给定则写到 ``<track_dir>/los_samp<iter>_misfit.png`` (与 .m 注释的命名思路一致).
    show
        若为 True 则 ``plt.show()`` (交互). 为 False 时只 ``savefig``(不在此强制 ``Agg``, 以免破坏同进程内后续 ``show_slip_model`` 的 GUI 后端).

    Returns
    -------
    out_figure
        已保存的 PNG 绝对路径(与自动命名规则一致时即 ``<track>/los_samp*\_misfit_iter*.png``).
    """
    sampled_data_file = os.path.abspath(str(sampled_data_file))
    if not os.path.isfile(sampled_data_file):
        raise FileNotFoundError(sampled_data_file)

    d = loadmat(sampled_data_file, squeeze_me=True, struct_as_record=False)
    sid = d.get("sampled_insar_data")
    if sid is None:
        raise KeyError("sampled_insar_data 不存在于 %s" % sampled_data_file)
    sid = np.asarray(sid, dtype=np.float64)
    if sid.ndim != 2 or sid.shape[1] < 6:
        raise ValueError("sampled_insar_data 期望 (N,>=6), 得 %r" % (sid.shape,))

    losin = np.asarray(sid[:, 2], dtype=np.float64).ravel()
    losm = np.asarray(los_model, dtype=np.float64).ravel()
    if losm.size != losin.size:
        raise ValueError("los_model 长度 %d 与 sampled 观测点数 %d 不一致" % (losm.size, losin.size))
    los_res = losin - losm
    # x, y: m -> km, 与 .m: xin = (:,1)/1000, yin = (:,2)/1000
    xin = (sid[:, 0] / 1000.0).ravel()
    yin = (sid[:, 1] / 1000.0).ravel()
    look_angle = sid[:, 3:6]

    filepath, fname_full = os.path.split(sampled_data_file)
    save_stem, _ = os.path.splitext(fname_full)
    label_name = filepath

    defo_min = -float(defo_max)
    res_max = float(misfit_range)

    fault_path = str(fault).strip() if fault is not None else ""
    seg: Optional[np.ndarray] = None
    if fault_path and os.path.isfile(fault_path):
        seg = np.loadtxt(fault_path, dtype=np.float64)
        if seg.ndim == 1:
            seg = seg.reshape(1, -1)
        if seg.shape[1] < 4:
            raise ValueError("fault 文件每行需至少 4 列, 得 shape=%s" % (seg.shape,))

    x0_ref, y0_ref = _ll2xy(lonc, latc, ref_lon)

    def _plot_fault_lines(ax: Any) -> None:
        if seg is None or seg.size == 0:
            return
        lonf = np.concatenate([seg[:, 0], seg[:, 2]])
        latf = np.concatenate([seg[:, 1], seg[:, 3]])
        nseg = int(lonf.size // 2)
        for ii in range(nseg):
            slon = np.array([lonf[ii], lonf[ii + nseg]], dtype=np.float64)
            slat = np.array([latf[ii], latf[ii + nseg]], dtype=np.float64)
            xx, yy = _ll2xy(slon, slat, ref_lon)
            xs = (np.asarray(xx) - x0_ref) / 1000.0
            ys = (np.asarray(yy) - y0_ref) / 1000.0
            ax.plot(
                np.ravel(xs, order="C"), np.ravel(ys, order="C"),
                c="k", linewidth=1.5,
            )

    import matplotlib
    import matplotlib.pyplot as plt
    # jet 与 .m 一致 (不强切 Agg, 避免影响同进程后续 show_slip_model 的交互后端)
    try:
        cmap = matplotlib.colormaps["jet"]
    except (AttributeError, KeyError, TypeError):
        from matplotlib import cm
        cmap = cm.get_cmap("jet")  # type: ignore[assignment]
    fig = plt.figure(figsize=(10, 7.0))
    sz = 30

    ax1 = fig.add_axes([0.04, 0.55, 0.42, 0.42])
    ax1.set_facecolor("0.95")
    sc1 = ax1.scatter(xin, yin, c=losin, cmap=cmap, s=sz)
    plt.colorbar(sc1, ax=ax1, shrink=0.7)
    ax1.set_title("Sampled Data (%s)" % label_name, fontsize=10)
    sc1.set_clim(defo_min, float(defo_max))
    ax1.set_aspect("equal", adjustable="box")
    if axis_range is not None and len(axis_range) == 4:
        ax1.set_xlim(float(axis_range[0]), float(axis_range[1]))
        ax1.set_ylim(float(axis_range[2]), float(axis_range[3]))
    _plot_fault_lines(ax1)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_axes([0.54, 0.55, 0.42, 0.42])
    sc2 = ax2.scatter(xin, yin, c=losm, cmap=cmap, s=sz)
    plt.colorbar(sc2, ax=ax2, shrink=0.7)
    ax2.set_title("Model", fontsize=10)
    sc2.set_clim(defo_min, float(defo_max))
    ax2.set_aspect("equal", adjustable="box")
    if axis_range is not None and len(axis_range) == 4:
        ax2.set_xlim(float(axis_range[0]), float(axis_range[1]))
        ax2.set_ylim(float(axis_range[2]), float(axis_range[3]))
    _plot_fault_lines(ax2)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_axes([0.04, 0.08, 0.42, 0.40])
    sc3 = ax3.scatter(xin, yin, c=los_res, cmap=cmap, s=sz)
    plt.colorbar(sc3, ax=ax3, shrink=0.7)
    ax3.set_title("Residual", fontsize=10)
    sc3.set_clim(-res_max, res_max)
    ax3.set_aspect("equal", adjustable="box")
    if axis_range is not None and len(axis_range) == 4:
        ax3.set_xlim(float(axis_range[0]), float(axis_range[1]))
        ax3.set_ylim(float(axis_range[2]), float(axis_range[3]))
    _plot_fault_lines(ax3)
    ax3.grid(True, alpha=0.3)

    rms0 = float(np.sum(losin * losin))
    rms1 = float(np.sum(los_res * los_res))
    redu = 100.0 * (rms0 - rms1) / rms0 if rms0 else 0.0
    txt = "rms = %.6e\nmisfit =  %.6e\n(dat., res.) =(%.4f%%)" % (rms0, rms1, redu)
    fig.text(0.75, 0.25, txt, ha="center", va="center", fontsize=12, family="monospace")

    if out_figure is None:
        out_figure = os.path.join(
            filepath, "%s_misfit_iter%d.png" % (save_stem, int(iter_step))
        )
    out_figure = str(out_figure)
    fig.patch.set_facecolor("w")
    fig.savefig(out_figure, dpi=200, bbox_inches="tight", facecolor="w")

    if save_los_model_mat:
        sampled_model = np.ascontiguousarray(
            np.column_stack(
                [xin * 1000.0, yin * 1000.0, np.asarray(losm, dtype=np.float64), look_angle]
            ),
            dtype=np.float64,
        )
        mname = "los_model.mat" if str(model_type).lower() == "okada" else "los_model_layered.mat"
        out_model = os.path.join(filepath, mname)
        savemat(
            out_model, {"sampled_model": sampled_model},
            do_compression=True,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_figure
