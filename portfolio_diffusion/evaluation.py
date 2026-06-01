from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def acf1_numpy_centered(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute lag-1 autocorrelation per asset after demeaning each asset.

    Parameters
    ----------
    x:
        Array with shape [time, num_assets].

    Returns
    -------
    np.ndarray
        Lag-1 autocorrelation for each asset, shape [num_assets].
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must have shape [time, num_assets], got {x.shape}")
    if x.shape[0] < 2:
        return np.full(x.shape[1], np.nan)

    centered = x - x.mean(axis=0, keepdims=True)
    num = (centered[:-1] * centered[1:]).mean(axis=0)
    den = (centered * centered).mean(axis=0) + eps
    return num / den


def generated_path_diagnostics(
    real_series: np.ndarray,
    generated_paths: np.ndarray,
    asset_names=None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compare real test returns against generated autoregressive paths.

    Parameters
    ----------
    real_series:
        Real returns with shape [T, D].
    generated_paths:
        Generated returns with shape [T, S, D], where S is number of paths.
    asset_names:
        Optional list of length D.

    Returns
    -------
    diagnostics_df, summary
        diagnostics_df has per-asset real/generated mean, std, and ACF1.
        summary contains MAE metrics across assets.
    """
    real_series = np.asarray(real_series, dtype=np.float64)
    generated_paths = np.asarray(generated_paths, dtype=np.float64)

    if real_series.ndim != 2:
        raise ValueError(f"real_series must have shape [T, D], got {real_series.shape}")
    if generated_paths.ndim != 3:
        raise ValueError(f"generated_paths must have shape [T, S, D], got {generated_paths.shape}")
    if generated_paths.shape[0] != real_series.shape[0] or generated_paths.shape[2] != real_series.shape[1]:
        raise ValueError(
            "generated_paths must match real_series in time and assets: "
            f"real={real_series.shape}, generated={generated_paths.shape}"
        )

    T, S, D = generated_paths.shape
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(D)]

    gen_flat = generated_paths.reshape(T * S, D)

    real_mean = real_series.mean(axis=0)
    gen_mean = gen_flat.mean(axis=0)

    real_std = real_series.std(axis=0, ddof=0)
    gen_std = gen_flat.std(axis=0, ddof=0)

    real_acf1 = acf1_numpy_centered(real_series)
    gen_acf1_by_path = np.stack(
        [acf1_numpy_centered(generated_paths[:, s, :]) for s in range(S)],
        axis=0,
    )
    gen_acf1 = np.nanmean(gen_acf1_by_path, axis=0)

    diagnostics_df = pd.DataFrame(
        {
            "asset": list(asset_names),
            "real_mean": real_mean,
            "gen_mean": gen_mean,
            "mean_abs_error": np.abs(real_mean - gen_mean),
            "real_std": real_std,
            "gen_std": gen_std,
            "std_abs_error": np.abs(real_std - gen_std),
            "real_acf1": real_acf1,
            "gen_acf1": gen_acf1,
            "acf1_abs_error": np.abs(real_acf1 - gen_acf1),
        }
    )

    summary = {
        "Mean MAE": float(diagnostics_df["mean_abs_error"].mean()),
        "Std MAE": float(diagnostics_df["std_abs_error"].mean()),
        "ACF1 MAE": float(diagnostics_df["acf1_abs_error"].mean()),
    }

    return diagnostics_df, summary


def correlation_mae(real_series: np.ndarray, generated_paths: np.ndarray) -> float:
    """
    Mean absolute difference between real and average generated correlation matrices.

    generated_paths should have shape [T, S, D]. The generated correlation matrix
    is computed for each generated path and then averaged across paths.
    """
    real_series = np.asarray(real_series, dtype=np.float64)
    generated_paths = np.asarray(generated_paths, dtype=np.float64)

    real_corr = np.corrcoef(real_series.T)
    gen_corrs = np.stack(
        [np.corrcoef(generated_paths[:, s, :].T) for s in range(generated_paths.shape[1])],
        axis=0,
    )
    gen_corr = np.nanmean(gen_corrs, axis=0)

    mask = ~np.eye(real_corr.shape[0], dtype=bool)
    return float(np.nanmean(np.abs(real_corr[mask] - gen_corr[mask])))
