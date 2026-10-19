"""Bounded linear least squares (MATLAB ``lsqlin`` equivalent).

Shared by ``make_fault_from_insar1``, ``make_rms``, checkerboard, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, lsq_linear


@dataclass
class SolveResult:
    u: np.ndarray
    residual: np.ndarray
    exitflag: int
    resnorm: float
    name: str
    sub: str
    message: str = ""
    nit: Optional[int] = None
    success: bool = False

    @property
    def summary(self) -> str:
        if self.name == "lsq_linear":
            return "lsq_linear(%s) nit=%s %s" % (self.sub, self.nit, self.message)
        return "least_squares success=%s %s" % (self.success, self.message)


def solve_bounded_lsq(
    Greens: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_iter: int = 100,
    *,
    verbose: bool = True,
) -> SolveResult:
    """
    min ||Greens * u - b||^2  s.t.  lb <= u <= ub

    Tries ``lsq_linear`` with ``lsq_solver='exact'`` then ``'lsmr'``;
    falls back to ``least_squares`` (TRF).
    """
    Greens_ = np.ascontiguousarray(Greens, dtype=np.float64)
    b_ = np.ascontiguousarray(b, dtype=np.float64).ravel()
    lb_ = np.ascontiguousarray(lb, dtype=np.float64).ravel()
    ub_ = np.ascontiguousarray(ub, dtype=np.float64).ravel()
    n_iter = int(max_iter)

    res_lsq = None
    lsq_sub = "exact"
    for solver in ("exact", "lsmr"):
        try:
            res_lsq = lsq_linear(
                Greens_, b_, bounds=(lb_, ub_), method="trf",
                tol=1e-12, lsq_solver=solver, max_iter=n_iter, verbose=0,
                lsmr_tol=1e-12,
                lsmr_maxiter=min(10_000, 50_000 * Greens_.shape[1]),
            )
            lsq_sub = solver
            if res_lsq.success or res_lsq.status in (1, 2, 3, 4):
                break
        except (np.linalg.LinAlgError, ValueError, RuntimeError, MemoryError) as e:
            if verbose:
                print(
                    "  lsq_linear (lsq_solver=%s) failed: %r — try next" % (solver, e),
                    flush=True,
                )
            res_lsq = None
            continue

    if res_lsq is not None and (res_lsq.success or res_lsq.status in (1, 2, 3, 4)):
        u = np.asarray(res_lsq.x, dtype=np.float64).ravel()
        resnorm = 2.0 * float(res_lsq.cost)
        residual = res_lsq.fun
        if residual is None or not hasattr(residual, "shape"):
            residual = Greens_ @ u - b_
        residual = np.asarray(residual, dtype=np.float64).ravel()
        exitflag = 1 if res_lsq.success else (
            0 if getattr(res_lsq, "status", -1) == 0 else -1
        )
        if verbose and res_lsq.nit is not None:
            print(
                "  solver: lsq_linear  sub=%s  nit=%s  %s"
                % (lsq_sub, res_lsq.nit, res_lsq.message),
                flush=True,
            )
        return SolveResult(
            u=u,
            residual=residual,
            exitflag=exitflag,
            resnorm=resnorm,
            name="lsq_linear",
            sub=lsq_sub,
            message=str(res_lsq.message),
            nit=res_lsq.nit,
            success=bool(res_lsq.success),
        )

    if verbose and res_lsq is not None:
        print(
            "  lsq_linear not acceptable (status=%s) — fallback least_squares"
            % getattr(res_lsq, "status", "?"),
            flush=True,
        )
    elif verbose:
        print("  lsq_linear 不可用 — 兜底 least_squares(TRF)", flush=True)

    x0 = np.clip(np.zeros(Greens_.shape[1]), lb_, ub_)

    def _fun(x):
        return Greens_ @ x - b_

    res = least_squares(
        _fun, x0, jac=lambda x: Greens_, bounds=(lb_, ub_),
        method="trf", tr_solver="lsmr",
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=max(500, n_iter * 20), verbose=0,
    )
    u = np.asarray(res.x, dtype=np.float64).ravel()
    residual = Greens_ @ u - b_
    resnorm = (
        2.0 * float(res.cost) if res.cost is not None
        else float(np.sum(residual ** 2))
    )
    if verbose:
        print(
            "  solver: least_squares  sub=trf  success=%s  %s"
            % (getattr(res, "success", False), getattr(res, "message", "")),
            flush=True,
        )
    return SolveResult(
        u=u,
        residual=np.asarray(residual, dtype=np.float64).ravel(),
        exitflag=1 if res.success else -1,
        resnorm=resnorm,
        name="least_squares",
        sub="trf",
        message=str(getattr(res, "message", "")),
        nit=getattr(res, "nfev", None),
        success=bool(res.success),
    )


def solve_bounded_lsq_u(
    Greens: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_nfev: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, str]:
    """Compatibility wrapper returning ``(u, summary)`` for make_rms-style callers."""
    r = solve_bounded_lsq(Greens, b, lb, ub, max_iter=max_nfev, verbose=verbose)
    return r.u, r.summary
