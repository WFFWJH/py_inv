"""Port of bounds_new.m."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def bounds_new(NS: int,
               NT: int,
               tSm: Sequence[int],
               add_col: int,
               Con: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Construct lb, ub vectors.

    Parameters
    ----------
    NS : int
        number of fault segments
    NT : int
        degrees of freedom per patch (typically 2: strike-slip + dip-slip)
    tSm : array-like (len Nf+1, first element is 0)
        number of patches in each segment, prepended with 0 (cumulative helper).
    add_col : int
        number of ramp coefficient columns appended.
    Con : sequence of 3 ints
        Sign constraint [strike, dip, normal]: +1 positivity, -1 negativity, 0 none.
    """
    slip_max = 10  # 1000 cm
    tSm = np.asarray(tSm, dtype=int)
    Npatch = int(tSm.sum())

    lb = -slip_max * np.ones(NT * Npatch, dtype=np.float64)
    ub = slip_max * np.ones(NT * Npatch, dtype=np.float64)
    # (the MATLAB script also re-asserts slip bounds on the dip block, same value)

    NS1 = NS
    Con = list(Con)

    for i in range(1, NS1 + 1):
        # MATLAB: k1 = sum(tSm(1:i))+1; k2 = sum(tSm(1:i+1));
        k1 = int(tSm[:i].sum())          # 0-based inclusive start
        k2 = int(tSm[:i + 1].sum())      # 0-based exclusive end
        for k in range(k1, k2):
            if Con[0] > 0: lb[k] = 0.0
            if Con[1] > 0: lb[k + Npatch] = 0.0
            if Con[0] < 0: ub[k] = 0.0
            if Con[1] < 0: ub[k + Npatch] = 0.0

    # Ramp coefficients have no bound
    if add_col > 0:
        ramp_lb = -np.inf * np.ones(add_col, dtype=np.float64)
        ramp_ub = np.inf * np.ones(add_col, dtype=np.float64)
        lb = np.concatenate([lb, ramp_lb])
        ub = np.concatenate([ub, ramp_ub])

    return lb, ub
