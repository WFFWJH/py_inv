"""Workflow configuration: configfile.txt + configpara.txt + data_list."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

@dataclass
class WorkflowConfig:
    """流程全部可调参数; 由 ``load_workflow_config`` 从两个 txt 合并填充."""

    # --- configfile.txt ---
    data_list: str = "datav0.list"
    fault_file: str = "fault_trace.txt"
    segment_file: str = "seg_connect"

    # --- configpara.txt ---
    lonc: float = 95.33
    latc: float = 19.61
    ref_lon: float = 95.0
    iter_step: int = 0
    iter_step2: int = 1
    con: Tuple[int, int, int] = (0, 0, 0)
    dip: List[float] = field(default_factory=lambda: [82.0, 82.0, 82.0, 82.0])
    ramp_choice: str = "qu_ramp_7"
    w_ratio: float = 1.1
    l_ratio: float = 1.2
    width: float = 20e3
    len_top: float = 2e3
    layers: int = 5
    default_region: Tuple[float, float, float, float] = (95.0, 97.52, 15.17, 24.05)
    okada_backend: str = "numpy"
    max_nfev: int = 100
    model_type: str = "okada"


def _parse_keyed_txt(path: str) -> Dict[str, List[str]]:
    """解析 ``##key`` / 值行格式; 以 ``#`` 开头的行视为注释."""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("##") and not raw.startswith("###"):
                key_parts = raw[2:].strip().split()
                if not key_parts:
                    raise ValueError(f"空配置键: {path}")
                current = key_parts[0].lower()
                sections.setdefault(current, [])
                continue
            if raw.startswith("#") or current is None:
                continue
            sections[current].append(raw)
    return sections


def _first_str(sec: Dict[str, List[str]], key: str, default: str) -> str:
    vals = sec.get(key) or []
    if not vals:
        return default
    if len(vals) != 1 or len(vals[0].split()) != 1:
        raise ValueError(f"配置键 {key!r} 必须且只能有一个值")
    return vals[0].strip()


def _floats(sec: Dict[str, List[str]], key: str) -> List[float]:
    out: List[float] = []
    for line in sec.get(key) or []:
        for tok in line.split():
            try:
                out.append(float(tok))
            except ValueError as exc:
                raise ValueError(f"配置键 {key!r} 含非法数值 {tok!r}") from exc
    return out


def _optional_float(sec: Dict[str, List[str]], key: str, default: float) -> float:
    nums = _floats(sec, key)
    if len(nums) > 1:
        raise ValueError(f"配置键 {key!r} 只能有一个数值")
    return float(nums[0]) if nums else default


def _optional_int(sec: Dict[str, List[str]], key: str, default: int) -> int:
    nums = _floats(sec, key)
    if len(nums) > 1:
        raise ValueError(f"配置键 {key!r} 只能有一个整数")
    if nums and (not np.isfinite(nums[0]) or not nums[0].is_integer()):
        raise ValueError(f"配置键 {key!r} 必须是有限整数")
    return int(nums[0]) if nums else default


def _reject_unknown_keys(
    sections: Dict[str, List[str]], allowed: Sequence[str], filename: str
) -> None:
    unknown = sorted(set(sections) - set(allowed))
    if unknown:
        raise ValueError(
            "%s 含未知配置键: %s" % (filename, ", ".join(unknown))
        )


def load_workflow_config(
    config_dir: Optional[Union[str, os.PathLike]] = None,
) -> Tuple[str, WorkflowConfig]:
    """
    从目录读取 ``configfile.txt`` + ``configpara.txt``, 合并为 ``WorkflowConfig``.

    返回 ``(root_abs, config)``. 缺文件或缺必需键时抛错; 可选键用 dataclass 默认值.
    """
    # Default config dir = project root (parent of this package)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(pkg_dir)
    if config_dir is not None:
        root = os.path.abspath(config_dir)
    else:
        cwd = os.getcwd()
        has_config = any(
            os.path.isfile(os.path.join(cwd, filename))
            for filename in ("configfile.txt", "configpara.txt")
        )
        root = cwd if has_config else project_root
    cfile = os.path.join(root, "configfile.txt")
    pfile = os.path.join(root, "configpara.txt")
    if not os.path.isfile(cfile) or not os.path.isfile(pfile):
        raise FileNotFoundError("需要 %s 与 %s" % (cfile, pfile))

    files = _parse_keyed_txt(cfile)
    paras = _parse_keyed_txt(pfile)
    base = WorkflowConfig()

    _reject_unknown_keys(
        files, ("data_list", "fault_file", "segment_file"), cfile
    )
    _reject_unknown_keys(
        paras,
        (
            "lonc", "latc", "ref_lon", "iter_step", "iter_step2", "con",
            "dip", "ramp_choice", "w_ratio", "l_ratio", "width", "len_top",
            "layers", "default_region", "okada_backend", "max_nfev",
            "model_type",
        ),
        pfile,
    )
    required_files = ("data_list", "fault_file", "segment_file")
    missing = [k for k in required_files if not (files.get(k))]
    if missing:
        raise ValueError("configfile.txt 缺少键或值: %s (%s)" % (", ".join(missing), cfile))

    con_nums = _floats(paras, "con")
    if con_nums and (
        len(con_nums) != 3
        or not np.all(np.isfinite(con_nums))
        or any(not value.is_integer() for value in con_nums)
    ):
        raise ValueError("configpara.txt 的 ##con 必须恰有 3 个数值")
    if con_nums:
        con = (int(con_nums[0]), int(con_nums[1]), int(con_nums[2]))
    else:
        con = base.con

    dip = _floats(paras, "dip") or list(base.dip)

    reg = _floats(paras, "default_region")
    if reg and len(reg) != 4:
        raise ValueError("configpara.txt 的 ##default_region 必须恰有 4 个数值")
    if reg:
        default_region = (reg[0], reg[1], reg[2], reg[3])
    else:
        default_region = base.default_region

    cfg = replace(
        base,
        data_list=_first_str(files, "data_list", base.data_list),
        fault_file=_first_str(files, "fault_file", base.fault_file),
        segment_file=_first_str(files, "segment_file", base.segment_file),
        lonc=_optional_float(paras, "lonc", base.lonc),
        latc=_optional_float(paras, "latc", base.latc),
        ref_lon=_optional_float(paras, "ref_lon", base.ref_lon),
        iter_step=_optional_int(paras, "iter_step", base.iter_step),
        iter_step2=_optional_int(paras, "iter_step2", base.iter_step2),
        con=con,
        dip=[float(x) for x in dip],
        ramp_choice=_first_str(paras, "ramp_choice", base.ramp_choice),
        w_ratio=_optional_float(paras, "w_ratio", base.w_ratio),
        l_ratio=_optional_float(paras, "l_ratio", base.l_ratio),
        width=_optional_float(paras, "width", base.width),
        len_top=_optional_float(paras, "len_top", base.len_top),
        layers=_optional_int(paras, "layers", base.layers),
        default_region=default_region,
        okada_backend=_first_str(paras, "okada_backend", base.okada_backend),
        max_nfev=_optional_int(paras, "max_nfev", base.max_nfev),
        model_type=_first_str(paras, "model_type", base.model_type),
    )
    numeric_values = (
        cfg.lonc, cfg.latc, cfg.ref_lon, cfg.w_ratio, cfg.l_ratio,
        cfg.width, cfg.len_top, *cfg.dip, *cfg.default_region,
    )
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("configpara.txt 中的坐标、几何参数和倾角必须为有限值")
    if not cfg.dip or any(angle <= 0.0 or angle > 180.0 for angle in cfg.dip):
        raise ValueError("##dip 中的角度必须满足 0 < dip <= 180")
    if cfg.layers < 1 or cfg.width <= 0 or cfg.len_top <= 0:
        raise ValueError("##layers、##width 和 ##len_top 必须为正")
    if cfg.w_ratio <= 0 or cfg.l_ratio <= 0:
        raise ValueError("##w_ratio 和 ##l_ratio 必须为正")
    xmin, xmax, ymin, ymax = cfg.default_region
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("##default_region 必须满足 xmin < xmax 且 ymin < ymax")
    if cfg.iter_step < 0 or cfg.iter_step2 < 0 or cfg.max_nfev < 1:
        raise ValueError("迭代标记不能为负，##max_nfev 必须为正")
    if cfg.okada_backend not in ("numpy", "numba", "auto"):
        raise ValueError("##okada_backend 必须是 numpy、numba 或 auto")
    if cfg.model_type != "okada":
        raise ValueError("当前仅支持 ##model_type=okada")
    if cfg.ramp_choice not in ("none", "bi_ramp", "qu_ramp_5", "qu_ramp_7"):
        raise ValueError(
            "##ramp_choice 必须是 none、bi_ramp、qu_ramp_5 或 qu_ramp_7"
        )
    return root, cfg


def read_data_list(
    data_list_path: str,
    *,
    default_region: Tuple[float, float, float, float] = (95.0, 97.52, 15.17, 24.05),
) -> Tuple[List[str], List[int], np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    每行 ``track_path npt [xmin xmax ymin ymax] type``.

    支持 2、3 或 7 列；未知类型和不完整区域会明确报错。
    """
    base = os.path.dirname(os.path.abspath(data_list_path))
    tracks: List[str] = []
    npt: List[int] = []
    regions: List[Tuple[float, float, float, float]] = []
    dtypes: List[str] = []
    with open(data_list_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.partition("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) not in (2, 3, 7):
                raise ValueError(
                    "%s:%d: 需要 2、3 或 7 列，实际为 %d 列"
                    % (data_list_path, line_number, len(parts))
                )
            tr_rel = parts[0]
            try:
                n_float = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    "%s:%d: npt 必须是正整数" % (data_list_path, line_number)
                ) from exc
            if not np.isfinite(n_float) or not n_float.is_integer() or n_float <= 0:
                raise ValueError(
                    "%s:%d: npt 必须是正整数" % (data_list_path, line_number)
                )
            n = int(n_float)
            if len(parts) == 7:
                reg = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                dtp = parts[6].lower()
            elif len(parts) == 3:
                reg = default_region
                dtp = parts[2].lower()
            else:
                reg = default_region
                dtp = "insar"
            if dtp == "rng":
                dtp = "insar"
            if dtp not in ("insar", "azi", "azo"):
                raise ValueError(
                    "%s:%d: 不支持的数据类型 %r (应为 insar/rng、azi 或 azo)"
                    % (data_list_path, line_number, dtp)
                )
            if not np.all(np.isfinite(reg)):
                raise ValueError(
                    "%s:%d: 区域边界必须为有限数值" % (data_list_path, line_number)
                )
            if reg[0] >= reg[1] or reg[2] >= reg[3]:
                raise ValueError(
                    "%s:%d: 区域必须满足 xmin < xmax 且 ymin < ymax"
                    % (data_list_path, line_number)
                )
            tr_abs = os.path.normpath(os.path.join(base, tr_rel))
            tracks.append(tr_abs)
            npt.append(n)
            regions.append(reg)
            dtypes.append(dtp)
    if not tracks:
        raise ValueError("data_list 中无有效道: %s" % data_list_path)
    region = np.asarray(regions, dtype=np.float64)
    nmin: List[int] = []
    nmax: List[int] = []
    for t in dtypes:
        if t == "insar":
            nmin.append(8)
            nmax.append(500)
        elif t == "azo":
            nmin.append(2)
            nmax.append(50)
        else:
            nmin.append(8)
            nmax.append(500)
    return (
        tracks,
        npt,
        region,
        dtypes,
        np.asarray(nmin, dtype=np.int64),
        np.asarray(nmax, dtype=np.int64),
    )


def count_fault_segments(fault_path: str) -> int:
    n = 0
    with open(fault_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


def resolve_under_root(root: str, p: str) -> str:
    p = p.strip()
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(root, p))


def resolve_geo_origin(
    ref_lon: Optional[float] = None,
    lonc: Optional[float] = None,
    latc: Optional[float] = None,
    config_dir: Optional[Union[str, os.PathLike]] = None,
) -> Tuple[float, float, float]:
    """把 ``ref_lon`` / ``lonc`` / ``latc`` 补齐.

    三者都给了就原样返回; 缺哪个就从 ``configpara.txt`` 补哪个 (同一个 key 名).
    ``config_dir=None`` 时用项目根目录 (``load_workflow_config`` 的默认行为).

    这样绘图函数不必再把 ``95.0 / 95.33 / 19.61`` 硬编码一遍 —— 那三个值本来就是
    反演配置里的, 抄一遍就容易和 ``configpara.txt`` 漂移.
    """
    if ref_lon is not None and lonc is not None and latc is not None:
        return float(ref_lon), float(lonc), float(latc)
    try:
        _root, cfg = load_workflow_config(config_dir)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "ref_lon/lonc/latc 未给全, 且无法从配置文件读取 (%s); "
            "请显式传入这三个参数, 或用 config_dir= 指定含 configpara.txt 的目录" % (exc,)
        ) from exc
    return (
        float(cfg.ref_lon if ref_lon is None else ref_lon),
        float(cfg.lonc if lonc is None else lonc),
        float(cfg.latc if latc is None else latc),
    )


_resolve_under_root = resolve_under_root
