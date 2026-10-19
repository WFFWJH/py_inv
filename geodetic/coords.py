"""Coordinate transforms: Transverse Mercator + 2-D rotation."""
from __future__ import annotations

import numpy as np


def ll2xy(xi, yi, lon_c):
    """Transverse Mercator (WGS84, k0=0.9996, x_0=5e5, y_0=0).

    Southern hemisphere returns negative y directly (matches pyproj's
    `+proj=tmerc +y_0=0` convention, no false-northing folding).
    """
    xi_arr = np.asarray(xi, dtype=np.float64)
    yi_arr = np.asarray(yi, dtype=np.float64)
    scalar_input = (xi_arr.ndim == 0) and (yi_arr.ndim == 0)
    xi_arr = np.atleast_1d(xi_arr)
    yi_arr = np.atleast_1d(yi_arr)

    r_dtor = np.pi / 180.0
    r_a = 6378137.0
    r_e2 = 0.006694379990141
    r_k0 = 0.9996
    r_lat0 = 0.0
    r_fe = 5e5
    r_fn = 0.0

    r_ep2 = r_e2 / (1.0 - r_e2)
    r_e4 = r_e2 * r_e2
    r_e6 = r_e4 * r_e2

    xi_rad = xi_arr * r_dtor
    yi_rad = yi_arr * r_dtor
    r_lon0 = lon_c * r_dtor

    sin_yi = np.sin(yi_rad)
    cos_yi = np.cos(yi_rad)
    tan_yi = np.tan(yi_rad)

    r_n = r_a / np.sqrt(1.0 - r_e2 * sin_yi * sin_yi)
    r_t = tan_yi * tan_yi
    r_t2 = r_t * r_t
    r_c = r_ep2 * cos_yi * cos_yi
    r_ba = (xi_rad - r_lon0) * cos_yi

    r_a2 = r_ba * r_ba
    r_a3 = r_ba * r_a2
    r_a4 = r_ba * r_a3
    r_a5 = r_ba * r_a4
    r_a6 = r_ba * r_a5

    r_m = r_a * (
        (1.0 - r_e2 / 4.0 - 3.0 * r_e4 / 64.0 - 5.0 * r_e6 / 256.0) * yi_rad
        - (3.0 * r_e2 / 8.0 + 3.0 * r_e4 / 32.0 + 45.0 * r_e6 / 1024.0) * np.sin(2.0 * yi_rad)
        + (15.0 * r_e4 / 256.0 + 45.0 * r_e6 / 1024.0) * np.sin(4.0 * yi_rad)
        - (35.0 * r_e6 / 3072.0) * np.sin(6.0 * yi_rad)
    )

    r_m0 = r_a * (
        (1.0 - r_e2 / 4.0 - 3.0 * r_e4 / 64.0 - 5.0 * r_e6 / 256.0) * r_lat0
        - (3.0 * r_e2 / 8.0 + 3.0 * r_e4 / 32.0 + 45.0 * r_e6 / 1024.0) * np.sin(2.0 * r_lat0)
        + (15.0 * r_e4 / 256.0 + 45.0 * r_e6 / 1024.0) * np.sin(4.0 * r_lat0)
        - (35.0 * r_e6 / 3072.0) * np.sin(6.0 * r_lat0)
    )

    xo = r_k0 * r_n * (
        r_ba
        + (1.0 - r_t + r_c) * r_a3 / 6.0
        + (5.0 - 18.0 * r_t + r_t2 + 72.0 * r_c - 58.0 * r_ep2) * r_a5 / 120.0
    )
    xo = xo + r_fe

    yo = r_k0 * (
        r_m
        - r_m0
        + r_n
        * tan_yi
        * (
            r_a2 / 2.0
            + (5.0 - r_t + 9.0 * r_c + 4.0 * r_c * r_c) * (r_a4 / 24.0)
            + (61.0 - 58.0 * r_t + r_t2 + 600.0 * r_c - 330.0 * r_ep2) * (r_a6 / 720.0)
        )
    )
    yo = yo + r_fn

    if scalar_input:
        return float(xo[0]), float(yo[0])
    return xo, yo



def xy2ll(xi, yi, lon_c):
    """Inverse of :func:`ll2xy`: projected TM coords -> lon/lat (degrees).

    Uses the same WGS84 ellipsoid, k0=0.9996, false easting 5e5, and
    y_0=0 convention as :func:`ll2xy` (no southern false northing).
    """
    xi_arr = np.asarray(xi, dtype=np.float64)
    yi_arr = np.asarray(yi, dtype=np.float64)
    scalar_input = (xi_arr.ndim == 0) and (yi_arr.ndim == 0)
    xi_arr = np.atleast_1d(xi_arr)
    yi_arr = np.atleast_1d(yi_arr)

    r_dtor = np.pi / 180.0
    r_a = 6378137.0
    r_e2 = 0.006694379990141
    r_k0 = 0.9996
    r_fe = 5e5

    r_ep2 = r_e2 / (1.0 - r_e2)
    r_e4 = r_e2 * r_e2
    r_e6 = r_e4 * r_e2

    r_vu_x = xi_arr - r_fe
    r_vu_y = yi_arr

    r_lon0 = lon_c * r_dtor

    r_et = np.sqrt(1.0 - r_e2)
    r_e1 = (1.0 - r_et) / (1.0 + r_et)
    r_e12 = r_e1 * r_e1
    r_e13 = r_e1 * r_e12
    r_e14 = r_e1 * r_e13

    r_m = r_vu_y / r_k0
    r_mu = r_m / (r_a * (1.0 - r_e2 / 4.0 - 3.0 * r_e4 / 64.0 - 5.0 * r_e6 / 256.0))
    r_lat1 = (
        r_mu
        + (3.0 * r_e1 / 2.0 - 27.0 * r_e13 / 32.0) * np.sin(2.0 * r_mu)
        + (21.0 * r_e12 / 16.0 - 55.0 * r_e14 / 32.0) * np.sin(4.0 * r_mu)
        + (51.0 * r_e13 / 96.0) * np.sin(6.0 * r_mu)
        + (1097.0 * r_e14 / 512.0) * np.sin(8.0 * r_mu)
    )

    sin_lat1 = np.sin(r_lat1)
    cos_lat1 = np.cos(r_lat1)
    tan_lat1 = np.tan(r_lat1)

    denom = np.sqrt(1.0 - r_e2 * sin_lat1 * sin_lat1)
    r_n = r_a / denom
    r_r = (r_a * (1.0 - r_e2)) / (denom**3)
    r_t = tan_lat1 * tan_lat1
    r_t2 = r_t * r_t
    r_c = r_ep2 * cos_lat1 * cos_lat1
    r_c2 = r_c * r_c

    r_d = r_vu_x / (r_n * r_k0)
    r_d2 = r_d * r_d
    r_d3 = r_d2 * r_d
    r_d4 = r_d3 * r_d
    r_d5 = r_d4 * r_d
    r_d6 = r_d5 * r_d

    yo = r_lat1 - (r_n * tan_lat1 / r_r) * (
        r_d2 / 2.0
        - (5.0 + 3.0 * r_t + 10.0 * r_c - 4.0 * r_c2 - 9.0 * r_ep2) * r_d4 / 24.0
        + (61.0 + 90.0 * r_t + 298.0 * r_c + 45.0 * r_t2 - 252.0 * r_ep2 - 3.0 * r_c2)
        * r_d6
        / 720.0
    )
    xo = r_lon0 + (
        r_d
        - (1.0 + 2.0 * r_t + r_c) * r_d3 / 6.0
        + (5.0 - 2.0 * r_c + 28.0 * r_t - 3.0 * r_c2 + 8.0 * r_ep2 + 24.0 * r_t2)
        * r_d5
        / 120.0
    ) / cos_lat1

    xo = xo / r_dtor
    yo = yo / r_dtor

    if scalar_input:
        return float(xo[0]), float(yo[0])
    return xo, yo


def xy2xy(x1, y1, phi):
    """2-D rotation by angle phi (radians). MATLAB ``xy2XY``."""
    c = np.cos(phi)
    s = np.sin(phi)
    return c * x1 + s * y1, -s * x1 + c * y1
