# -*- coding: utf-8 -*-
"""
属性转换脚本: Vs -> Vp, rho -> E, nu (含逆变换)
================================================

经验关系来源: Brocher (2005), BSSA 95(6): 2081-2092,
"Empirical Relations between Elastic Wavespeeds and Density in the Earth's Crust"

    Eq.(9)   Vp  = 0.9409 + 2.0947*Vs - 0.8206*Vs^2 + 0.2683*Vs^3 - 0.0251*Vs^4
             适用 Vs in [0, 4.5] km/s
    Eq.(1)   rho = 1.6612*Vp - 0.4721*Vp^2 + 0.0671*Vp^3 - 0.0043*Vp^4 + 0.000106*Vp^5
             (Nafe-Drake 曲线) 适用 Vp in [1.5, 8.5] km/s
    Eq.(10)  nu  = 0.5*[(Vp/Vs)^2 - 2] / [(Vp/Vs)^2 - 1]

各向同性线弹性转换:

    mu     = rho*Vs^2            (剪切模量)
    lambda = rho*Vp^2 - 2*mu     (Lame 第一参数)
    nu     = lambda / (2*(lambda + mu))
    E      = 2*mu*(1 + nu)       (杨氏模量)

单位约定(全脚本统一, 与论文一致):
    长度    : km            (x, y, depth)
    速度    : km/s          (Vs, Vp)
    密度    : g/cm^3        (rho)
    杨氏模量: GPa           (E; 因 rho[g/cm^3]*v^2[km/s^2] = 10^9 Pa, 数值上直接成立)
    泊松比  : 无量纲        (nu)

输入输出文件(绝对路径; 换机器/换目录测试时只需改下面 4 个常量):
    IN_3D / OUT_3D : 3D_Myanmar_H_v1.vel -> vs_with_props.txt
                     输出列: x(km) y(km) depth(km) Vs(km/s) Vp(km/s) rho(g/cm^3) E(GPa) nu
    IN_1D / OUT_1D : EW05.vel -> ew05.txt
                     输出列: depth(km) Vs(km/s) Vp(km/s) rho(g/cm^3)

注: ll2xy 按 UTM 标准返回米, convert_3d 中已换算为 km。
依赖: numpy
"""

import numpy as np

# ------------------ 文件路径(绝对路径) ------------------
IN_3D  = r"D:\Comsol_mphs\2622\2623_layer\3D_Myanmar_H_v1.vel"
OUT_3D = r"D:\Comsol_mphs\2622\2623_layer\vs_with_props.txt"
IN_1D  = r"D:\Comsol_mphs\2622\2623_layer\EW05.vel"
OUT_1D = r"D:\Comsol_mphs\2622\2623_layer\ew05.txt"

# 3D 流程: 相对坐标参考点与 UTM 中央经线
REF_LON, REF_LAT, LON_C = 95.33, 19.61, 95.0


# ------------------ Brocher (2005) 经验关系 ------------------
def empirical_vs(vs):
    """Vs(km/s) -> (Vp km/s, rho g/cm^3)   [Brocher 2005, Eq.(9) + Eq.(1)]

    注: 三次项系数 0.2683 为论文 Eq.(9) 原值(原 MATLAB 脚本误写为 0.2638)。
    """
    vs = np.asarray(vs, dtype=float)
    vp = 0.9409 + 2.0947*vs - 0.8206*vs**2 + 0.2683*vs**3 - 0.0251*vs**4
    rho = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5
    return vp, rho


# ------------------ 弹性常数转换 ------------------
def vpvsrho_to_Enu(vp, vs, rho):
    """(vp km/s, vs km/s, rho g/cm^3) -> (E GPa, nu)"""
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    mu = rho * vs**2                    # 剪切模量 (GPa)
    lam = rho * vp**2 - 2.0*mu          # Lame 第一参数 (GPa)
    denom = 2.0*(lam + mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = lam/denom
        E = 2.0*mu*(1.0 + nu)
    bad = (mu <= 0) | ~np.isfinite(mu) | (denom == 0) | ~np.isfinite(denom)
    if np.any(bad):
        print("[warn] vpvsrho_to_Enu: %d 个非物理输入(mu<=0 或除零), 已置 NaN"
              % int(np.sum(bad)))
        E = np.where(bad, np.nan, E)
        nu = np.where(bad, np.nan, nu)
    return E, nu


def Enu_rho_to_vpvs(E, nu, rho):
    """(E GPa, nu, rho g/cm^3) -> (vp km/s, vs km/s)   vpvsrho_to_Enu 的精确逆变换"""
    E = np.asarray(E, dtype=float)
    nu = np.asarray(nu, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if np.any((1.0 + nu == 0) | (1.0 - 2.0*nu == 0)):
        raise ValueError("nu = -1 或 0.5 会导致除零, 请检查输入")
    mu = E/(2.0*(1.0 + nu))
    lam = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    vs = np.sqrt(mu/rho)
    vp = np.sqrt((lam + 2.0*mu)/rho)
    return vp, vs


# ------------------ 经纬度 -> UTM (WGS84, k0=0.9996), 返回米 ------------------
def ll2xy(lon_deg, lat_deg, lon_c_deg):
    """(lon, lat, 中央经线 lon_c, 单位均为度) -> (x, y) UTM 坐标(米)"""
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    south = lat < 0
    lat = np.where(south, -lat, lat)          # 统一按北半球投影, 末尾再翻回负值

    d2r = np.pi/180.0
    xi, yi = lon*d2r, lat*d2r
    r_a, r_e2, r_k0, r_fe = 6378137.0, 0.006694379990141, 0.9996, 5.0e5
    r_ep2 = r_e2/(1.0 - r_e2)
    r_e4, r_e6 = r_e2**2, r_e2**3
    r_lon0 = lon_c_deg*d2r

    r_n = r_a/np.sqrt(1.0 - r_e2*np.sin(yi)**2)
    r_t = np.tan(yi)**2
    r_t2 = r_t**2
    r_c = r_ep2*np.cos(yi)**2
    r_ba = (xi - r_lon0)*np.cos(yi)
    r_a2, r_a3, r_a4, r_a5, r_a6 = r_ba**2, r_ba**3, r_ba**4, r_ba**5, r_ba**6
    r_m = r_a*((1.0 - r_e2/4.0 - 3.0*r_e4/64.0 - 5.0*r_e6/256.0)*yi
               - (3.0*r_e2/8.0 + 3.0*r_e4/32.0 + 45.0*r_e6/1024.0)*np.sin(2.0*yi)
               + (15.0*r_e4/256.0 + 45.0*r_e6/1024.0)*np.sin(4.0*yi)
               - (35.0*r_e6/3072.0)*np.sin(6.0*yi))

    x = r_k0*r_n*(r_ba + (1.0 - r_t + r_c)*r_a3/6.0
                  + (5.0 - 18.0*r_t + r_t2 + 72.0*r_c - 58.0*r_ep2)*r_a5/120.0) + r_fe
    y = r_k0*(r_m + r_n*np.tan(yi)*(r_a2/2.0
              + (5.0 - r_t + 9.0*r_c + 4.0*r_c**2)*r_a4/24.0
              + (61.0 - 58.0*r_t + r_t2 + 600.0*r_c - 330.0*r_ep2)*r_a6/720.0))
    y = np.where(south, -y, y)
    return x, y


# ------------------ 转换流程 ------------------
def convert_3d(infile=IN_3D, outfile=OUT_3D):
    """三维模型: (lon, lat, depth_km, Vs_km/s)
       -> x(km) y(km) depth(km) Vs(km/s) Vp(km/s) rho(g/cm^3) E(GPa) nu"""
    rows = []
    with open(infile, "r") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            p = line.split()
            if len(p) >= 4:
                rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
    d = np.array(rows)
    lon, lat, depth, vs_k = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

    idx = (depth < 51.0) & (vs_k > 0.0)
    lon, lat, depth, vs_k = lon[idx], lat[idx], depth[idx], vs_k[idx]

    x0, y0 = ll2xy(REF_LON, REF_LAT, LON_C)
    xu, yu = ll2xy(lon, lat, LON_C)
    x, y = (xu - x0[0])/1.0e3, (yu - y0[0])/1.0e3   # UTM 米 -> km

    vp_k, rho_g = empirical_vs(vs_k)
    E, nu = vpvsrho_to_Enu(vp_k, vs_k, rho_g)

    out = np.column_stack([x, y, depth, vs_k, vp_k, rho_g, E, nu])
    np.savetxt(outfile, out, fmt="%.15g", delimiter=" ")
    print("已写入 %s (%d 行)" % (outfile, len(out)))
    _report_3d(out)
    return out


def convert_1d(infile=IN_1D, outfile=OUT_1D):
    """一维模型: (depth_km, Vs_km/s) -> depth(km) Vs(km/s) Vp(km/s) rho(g/cm^3)"""
    d = np.loadtxt(infile)
    depth, vs_k = d[:, 0], d[:, 1]
    mask = depth < 70.0
    depth, vs_k = depth[mask], vs_k[mask]
    vp_k, rho_g = empirical_vs(vs_k)
    out = np.column_stack([depth, vs_k, vp_k, rho_g])
    np.savetxt(outfile, out, fmt="%.15g", delimiter="\t")
    print("已写入 %s (%d 行)" % (outfile, len(out)))
    _report_1d(out)
    return out


def _report_3d(out):
    x, y, dep = out[:, 0], out[:, 1], out[:, 2]
    vs, vp, rho, E, nu = out[:, 3], out[:, 4], out[:, 5], out[:, 6], out[:, 7]
    print("  x/y/depth : [%.1f, %.1f] / [%.1f, %.1f] / [%.2f, %.2f] km"
          % (np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y),
             np.nanmin(dep), np.nanmax(dep)))
    print("  Vs/Vp     : [%.4f, %.4f] / [%.4f, %.4f] km/s"
          % (np.nanmin(vs), np.nanmax(vs), np.nanmin(vp), np.nanmax(vp)))
    print("  rho       : [%.4f, %.4f] g/cm^3" % (np.nanmin(rho), np.nanmax(rho)))
    print("  E         : [%.4f, %.4f] GPa" % (np.nanmin(E), np.nanmax(E)))
    print("  nu        : [%.4f, %.4f]" % (np.nanmin(nu), np.nanmax(nu)))
    print("  提示: Vs>4.5 km/s(超出 Eq.9 拟合范围) %d 点; "
          "Vp<1.5 km/s(超出 Nafe-Drake 范围) %d 点; nu>=0.49 %d 点"
          % (int(np.sum(vs > 4.5)), int(np.sum(vp < 1.5)),
             int(np.sum(nu >= 0.49))))


def _report_1d(out):
    print("  Vs/Vp : [%.4f, %.4f] / [%.4f, %.4f] km/s"
          % (out[:, 1].min(), out[:, 1].max(), out[:, 2].min(), out[:, 2].max()))
    print("  rho   : [%.4f, %.4f] g/cm^3" % (out[:, 3].min(), out[:, 3].max()))
    print("  提示: Vs>4.5 km/s(超出 Eq.9 拟合范围) %d 点"
          % int(np.sum(out[:, 1] > 4.5)))


if __name__ == "__main__":
    print("=== 3D 模型转换 ===")
    convert_3d()
    print("=== 1D 模型转换 ===")
    convert_1d()
