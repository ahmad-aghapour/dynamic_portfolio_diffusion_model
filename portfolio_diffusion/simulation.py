from __future__ import annotations

import numpy as np


def simulate_arma_returns(
    T=6000,
    num_assets=5,
    ar=(0.35,),
    ma=(0.25,),
    sigma=0.01,
    corr=0.30,
    burn_in=500,
    seed=123,
):
    """
    Simulate multivariate ARMA return data.

    Output:
        returns: [T, num_assets]
    """
    rng = np.random.default_rng(seed)

    p = len(ar)
    q = len(ma)
    max_lag = max(p, q)

    total_T = T + burn_in

    cov = np.full((num_assets, num_assets), corr * sigma**2)
    np.fill_diagonal(cov, sigma**2)

    eps = rng.multivariate_normal(
        mean=np.zeros(num_assets),
        cov=cov,
        size=total_T,
    ).astype(np.float32)

    r = np.zeros((total_T, num_assets), dtype=np.float32)

    for t in range(max_lag, total_T):
        ar_part = np.zeros(num_assets, dtype=np.float32)
        ma_part = np.zeros(num_assets, dtype=np.float32)

        for i, phi in enumerate(ar, start=1):
            ar_part += phi * r[t - i]

        for j, theta in enumerate(ma, start=1):
            ma_part += theta * eps[t - j]

        r[t] = ar_part + eps[t] + ma_part

    return r[burn_in:]
