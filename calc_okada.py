import math
import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None


def calc_okada(HF, U, x, y, nu, delta, d, length, W, fault_type, strike, tp=None, backend="numpy"):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    scalar_input = (x_arr.ndim == 0) and (y_arr.ndim == 0)
    x_arr = np.atleast_1d(x_arr)
    y_arr = np.atleast_1d(y_arr)
    if tp is None:
        tp_arr = np.zeros_like(x_arr)
    else:
        tp_arr = np.atleast_1d(np.asarray(tp, dtype=np.float64))

    if backend == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is not available. Install numba or use backend='numpy'.")
        ux, uy, uz = _calc_okada_numba(HF, U, x_arr, y_arr, nu, delta, d, length, W, int(fault_type), strike, tp_arr)
    elif backend == "auto" and NUMBA_AVAILABLE:
        ux, uy, uz = _calc_okada_numba(HF, U, x_arr, y_arr, nu, delta, d, length, W, int(fault_type), strike, tp_arr)
    else:
        ux, uy, uz = _calc_okada_numpy(HF, U, x_arr, y_arr, nu, delta, d, length, W, int(fault_type), strike, tp_arr)

    if scalar_input:
        return float(ux[0]), float(uy[0]), float(uz[0])
    return ux, uy, uz


def _calc_okada_numpy(HF, U, x_arr, y_arr, nu, delta, d, length, W, fault_type, strike, tp_arr):
    cd = np.cos(delta)
    sd = np.sin(delta)
    d2 = d + W * sd + tp_arr
    x2 = x_arr - W * cd * np.cos(strike)
    y2 = y_arr + W * cd * np.sin(strike)
    strike2 = -strike + np.pi / 2.0
    cs = np.cos(strike2)
    sn = np.sin(strike2)
    rotx = x2 * cs + y2 * sn
    roty = -x2 * sn + y2 * cs
    L = length * 0.5
    a = 1.0 - 2.0 * nu
    const = -U / (2.0 * np.pi)
    p = roty * cd + d2 * sd
    q = roty * sd - d2 * cd
    cd_k = np.cos(delta)
    sd_k = np.sin(delta)
    epsn = 1.0e-15
    td_k = sd_k / cd_k if abs(cd_k) >= epsn else 0.0
    sd2_k = sd_k * sd_k
    cssd_k = cd_k * sd_k
    f1a, f2a, f3a = _fbi_fast_np(rotx + L, p, fault_type, q, a, cd_k, sd_k, td_k, sd2_k, cssd_k, epsn)
    f1b, f2b, f3b = _fbi_fast_np(rotx + L, p - W, fault_type, q, a, cd_k, sd_k, td_k, sd2_k, cssd_k, epsn)
    f1c, f2c, f3c = _fbi_fast_np(rotx - L, p, fault_type, q, a, cd_k, sd_k, td_k, sd2_k, cssd_k, epsn)
    f1d, f2d, f3d = _fbi_fast_np(rotx - L, p - W, fault_type, q, a, cd_k, sd_k, td_k, sd2_k, cssd_k, epsn)
    uxj = const * (f1a - f1b - f1c + f1d)
    uyj = const * (f2a - f2b - f2c + f2d)
    uz = const * (f3a - f3b - f3c + f3d)
    ux = HF * (-uyj * sn + uxj * cs)
    uy = HF * (uxj * sn + uyj * cs)
    return ux, uy, uz


def _fbi_fast_np(sig, eta, fault_type, q, a, cd, sd, td, sd2, cssd, epsn):
    sig2 = sig * sig
    eta2 = eta * eta
    q2 = q * q
    R = np.sqrt(sig2 + eta2 + q2)
    X = np.sqrt(sig2 + q2)
    ytil = eta * cd + q * sd
    dtil = eta * sd - q * cd
    Rdtil = R + dtil
    Rsig = R + sig
    Reta = R + eta
    RX = R + X
    lnRdtil = np.log(Rdtil)
    lnReta = np.log(Reta)
    ORRsig = 1.0 / (R * Rsig)
    ORReta = 1.0 / (R * Reta)
    OReta = 1.0 / Reta
    mask = np.abs(Reta) < epsn
    if np.any(mask):
        lnReta = lnReta.copy()
        OReta = OReta.copy()
        ORReta = ORReta.copy()
        lnReta[mask] = -np.log(R[mask] - eta[mask])
        OReta[mask] = 0.0
        ORReta[mask] = 0.0
    mask = np.abs(Rsig) < epsn
    if np.any(mask):
        ORRsig = ORRsig.copy()
        ORRsig[mask] = 0.0
    theta = np.zeros_like(sig, dtype=np.float64)
    mask = np.abs(q) > epsn
    if np.any(mask):
        theta[mask] = np.arctan((sig[mask] * eta[mask]) / (q[mask] * R[mask]))
    if abs(cd) < epsn:
        I5 = -a * sig * sd / Rdtil
        I4 = -a * q / Rdtil
        I3 = a / 2.0 * (eta / Rdtil + (ytil * q) / (Rdtil * Rdtil) - lnReta)
        I2 = -a * lnReta - I3
        I1 = -a / 2.0 * (sig * q) / (Rdtil * Rdtil)
    else:
        I5 = a * 2.0 / cd * np.arctan((eta * (X + q * cd) + X * RX * sd) / (sig * RX * cd))
        mask = np.abs(sig) < epsn
        if np.any(mask):
            I5 = I5.copy()
            I5[mask] = 0.0
        I4 = a / cd * (lnRdtil - sd * lnReta)
        I3 = a * (ytil / (cd * Rdtil) - lnReta) + td * I4
        I2 = -a * lnReta - I3
        I1 = -a / cd * sig / Rdtil - td * I5
    sigqORReta = sig * q * ORReta
    yqORReta = ytil * q * ORReta
    dqORReta = dtil * q * ORReta
    yqORRsig = ytil * q * ORRsig
    dqORRsig = dtil * q * ORRsig
    qOReta = q * OReta
    if fault_type == 1:
        f1 = sigqORReta + theta + I1 * sd
        f2 = yqORReta + (cd * qOReta) + I2 * sd
        f3 = dqORReta + (sd * qOReta) + I4 * sd
    elif fault_type == 2:
        f1 = q / R - I3 * cssd
        f2 = yqORRsig + cd * theta - I1 * cssd
        f3 = dqORRsig + sd * theta - I5 * cssd
    else:
        f1 = q2 * ORReta - I3 * sd2
        f2 = (-dqORRsig) - sd * (sigqORReta - theta) - I1 * sd2
        f3 = yqORRsig + cd * (sigqORReta - theta) - I5 * sd2
    return f1, f2, f3


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=False, nogil=True)
    def _fbi_scalar(sig, eta, q, a, cd, sd, td, sd2, cssd, epsn, fault_type):
        R = math.sqrt(sig * sig + eta * eta + q * q)
        X = math.sqrt(sig * sig + q * q)
        ytil = eta * cd + q * sd
        dtil = eta * sd - q * cd
        Rdtil = R + dtil
        Rsig = R + sig
        Reta = R + eta
        RX = R + X
        lnRdtil = math.log(Rdtil)
        if abs(Reta) < epsn:
            lnReta = -math.log(R - eta)
            OReta = 0.0
            ORReta = 0.0
        else:
            lnReta = math.log(Reta)
            OReta = 1.0 / Reta
            ORReta = 1.0 / (R * Reta)
        ORRsig = 0.0 if abs(Rsig) < epsn else 1.0 / (R * Rsig)
        theta = 0.0
        if abs(q) > epsn:
            theta = math.atan((sig * eta) / (q * R))
        if abs(cd) < epsn:
            I5 = -a * sig * sd / Rdtil
            I4 = -a * q / Rdtil
            I3 = a * 0.5 * (eta / Rdtil + (ytil * q) / (Rdtil * Rdtil) - lnReta)
            I2 = -a * lnReta - I3
            I1 = -a * 0.5 * (sig * q) / (Rdtil * Rdtil)
        else:
            den = sig * RX * cd
            I5 = a * 2.0 / cd * math.atan((eta * (X + q * cd) + X * RX * sd) / den)
            if abs(sig) < epsn:
                I5 = 0.0
            I4 = a / cd * (lnRdtil - sd * lnReta)
            I3 = a * (ytil / (cd * Rdtil) - lnReta) + td * I4
            I2 = -a * lnReta - I3
            I1 = -a / cd * sig / Rdtil - td * I5
        sigqORReta = sig * q * ORReta
        yqORReta = ytil * q * ORReta
        dqORReta = dtil * q * ORReta
        yqORRsig = ytil * q * ORRsig
        dqORRsig = dtil * q * ORRsig
        qOReta = q * OReta
        if fault_type == 1:
            f1 = sigqORReta + theta + I1 * sd
            f2 = yqORReta + (cd * qOReta) + I2 * sd
            f3 = dqORReta + (sd * qOReta) + I4 * sd
        elif fault_type == 2:
            f1 = q / R - I3 * cssd
            f2 = yqORRsig + cd * theta - I1 * cssd
            f3 = dqORRsig + sd * theta - I5 * cssd
        else:
            f1 = q * q * ORReta - I3 * sd2
            f2 = (-dqORRsig) - sd * (sigqORReta - theta) - I1 * sd2
            f3 = yqORRsig + cd * (sigqORReta - theta) - I5 * sd2
        return f1, f2, f3

    @njit(cache=True, fastmath=False, nogil=True)
    def _calc_okada_numba(HF, U, x_arr, y_arr, nu, delta, d, length, W, fault_type, strike, tp_arr):
        n = x_arr.size
        ux = np.empty(n, dtype=np.float64)
        uy = np.empty(n, dtype=np.float64)
        uz = np.empty(n, dtype=np.float64)
        cd = math.cos(delta)
        sd = math.sin(delta)
        cs = math.cos(-strike + math.pi / 2.0)
        sn = math.sin(-strike + math.pi / 2.0)
        xoff = W * cd * math.cos(strike)
        yoff = W * cd * math.sin(strike)
        L = length * 0.5
        a = 1.0 - 2.0 * nu
        const = -U / (2.0 * math.pi)
        epsn = 1.0e-15
        td = sd / cd if abs(cd) >= epsn else 0.0
        sd2 = sd * sd
        cssd = cd * sd
        for i in range(n):
            x2 = x_arr[i] - xoff
            y2 = y_arr[i] + yoff
            d2 = d + W * sd + tp_arr[i]
            rotx = x2 * cs + y2 * sn
            roty = -x2 * sn + y2 * cs
            p = roty * cd + d2 * sd
            q = roty * sd - d2 * cd
            f1a, f2a, f3a = _fbi_scalar(rotx + L, p, q, a, cd, sd, td, sd2, cssd, epsn, fault_type)
            f1b, f2b, f3b = _fbi_scalar(rotx + L, p - W, q, a, cd, sd, td, sd2, cssd, epsn, fault_type)
            f1c, f2c, f3c = _fbi_scalar(rotx - L, p, q, a, cd, sd, td, sd2, cssd, epsn, fault_type)
            f1d, f2d, f3d = _fbi_scalar(rotx - L, p - W, q, a, cd, sd, td, sd2, cssd, epsn, fault_type)
            uxj = const * (f1a - f1b - f1c + f1d)
            uyj = const * (f2a - f2b - f2c + f2d)
            uz[i] = const * (f3a - f3b - f3c + f3d)
            ux[i] = HF * (-uyj * sn + uxj * cs)
            uy[i] = HF * (uxj * sn + uyj * cs)
        return ux, uy, uz
else:
    def _calc_okada_numba(HF, U, x_arr, y_arr, nu, delta, d, length, W, fault_type, strike, tp_arr):
        raise RuntimeError("Numba is not available.")
