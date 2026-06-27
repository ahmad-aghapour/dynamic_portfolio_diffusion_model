from __future__ import annotations

import numpy as np
import torch
import pandas as pd


def acf1_numpy_centered(x: np.ndarray) -> np.ndarray:
    """Centered lag-1 autocorrelation for a [T, D] array."""
    x = np.asarray(x)
    x = x - x.mean(axis=0, keepdims=True)
    num = (x[:-1] * x[1:]).mean(axis=0)
    den = (x * x).mean(axis=0) + 1e-8
    return num / den


def acf1_torch_scenarios_centered(x: torch.Tensor) -> torch.Tensor:
    """Centered lag-1 autocorrelation for generated paths [horizon, num_samples, D]."""
    x = x - x.mean(dim=(0, 1), keepdim=True)
    x0 = x[:-1]
    x1 = x[1:]
    num = (x0 * x1).mean(dim=(0, 1))
    den = (x * x).mean(dim=(0, 1)) + 1e-8
    return num / den


def arma_distribution_diagnostics(
    real_standardized: np.ndarray,
    generated_standardized: torch.Tensor | np.ndarray,
    asset_names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Match the diagnostics in the original ARMA notebook.

    Parameters
    ----------
    real_standardized:
        Validation returns, shape [T_valid, D].
    generated_standardized:
        Generated returns, shape [horizon, num_samples, D].
    """
    if isinstance(generated_standardized, np.ndarray):
        gen_t = torch.as_tensor(generated_standardized, dtype=torch.float32)
    else:
        gen_t = generated_standardized.detach().cpu().float()

    real = np.asarray(real_standardized, dtype=np.float32)
    D = real.shape[1]
    if asset_names is None:
        asset_names = [f"Asset_{i+1}" for i in range(D)]

    real_mean = real.mean(axis=0)
    real_std = real.std(axis=0)
    real_acf1 = acf1_numpy_centered(real)

    gen_flat = gen_t.reshape(-1, gen_t.shape[-1])
    gen_mean = gen_flat.mean(dim=0).numpy()
    gen_std = gen_flat.std(dim=0).numpy()
    gen_acf1 = acf1_torch_scenarios_centered(gen_t).numpy()

    df = pd.DataFrame({
        "asset": asset_names,
        "real_mean": real_mean,
        "gen_mean": gen_mean,
        "abs_mean_error": np.abs(real_mean - gen_mean),
        "real_std": real_std,
        "gen_std": gen_std,
        "abs_std_error": np.abs(real_std - gen_std),
        "real_acf1": real_acf1,
        "gen_acf1": gen_acf1,
        "abs_acf1_error": np.abs(real_acf1 - gen_acf1),
    })

    summary = {
        "Mean MAE": float(df["abs_mean_error"].mean()),
        "Std MAE": float(df["abs_std_error"].mean()),
        "ACF1 MAE": float(df["abs_acf1_error"].mean()),
    }
    return df, summary

def historical_mean_covariance_for_dates(
    monthly_returns_df: pd.DataFrame,
    dates: pd.DatetimeIndex | pd.Index,
    lookback: int = 120,
    cov_shrinkage: float = 0.0,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate rolling historical mean/covariance before each target date.

    Returns means [T, N], covariances [T, N, N], and observations_used [T].
    """
    panel = monthly_returns_df.copy()
    panel.index = pd.DatetimeIndex(panel.index)
    aligned_dates = pd.DatetimeIndex(dates)
    full_raw = panel.values.astype(np.float64)

    means = []
    covariances = []
    observations_used = []

    for date in aligned_dates:
        pos = panel.index.get_loc(date)
        start = max(0, pos - lookback)
        hist = full_raw[start:pos]
        if hist.shape[0] < 2:
            raise ValueError(f"Need at least 2 historical rows before {date}.")

        mu = hist.mean(axis=0)
        cov = np.cov(hist, rowvar=False)
        cov = 0.5 * (cov + cov.T)

        if cov_shrinkage > 0.0:
            diag_cov = np.diag(np.diag(cov))
            cov = (1.0 - cov_shrinkage) * cov + cov_shrinkage * diag_cov

        cov = cov + ridge * np.eye(cov.shape[0])

        means.append(mu)
        covariances.append(cov)
        observations_used.append(hist.shape[0])

    return (
        np.asarray(means),
        np.asarray(covariances),
        np.asarray(observations_used),
    )


def generated_scenario_diagnostics(
    backtest_results: dict,
    monthly_returns_df: pd.DataFrame,
    lookback: int = 120,
    cov_shrinkage: float = 0.0,
    asset_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare generated scenarios with realized returns and rolling history.

    The diagnostics answer three questions:
      - How far is generated conditional mean from realized next return?
      - How far is generated covariance from rolling historical covariance?
      - Is generated volatility calibrated to realized absolute return scale?
    """
    scenarios = np.asarray(backtest_results["scenarios_raw"], dtype=np.float64)
    actual = np.asarray(backtest_results["actual_next_returns"], dtype=np.float64)
    dates = pd.DatetimeIndex(backtest_results["dates"])

    if scenarios.ndim != 3:
        raise ValueError(f"scenarios_raw must have shape [T, S, N], got {scenarios.shape}")
    if actual.shape != (scenarios.shape[0], scenarios.shape[2]):
        raise ValueError(
            "actual_next_returns must have shape [T, N] matching scenarios_raw."
        )

    T, _, N = scenarios.shape
    if asset_names is None:
        asset_names = [f"Asset_{i + 1}" for i in range(N)]

    gen_mean = scenarios.mean(axis=1)
    gen_cov = np.asarray([np.cov(scenarios[t], rowvar=False) for t in range(T)])
    gen_vol = scenarios.std(axis=1, ddof=1)

    hist_mean, hist_cov, obs_used = historical_mean_covariance_for_dates(
        monthly_returns_df=monthly_returns_df,
        dates=dates,
        lookback=lookback,
        cov_shrinkage=cov_shrinkage,
    )
    hist_vol = np.sqrt(np.clip(np.diagonal(hist_cov, axis1=1, axis2=2), 0.0, None))

    realized_abs = np.abs(actual)
    cov_fro_error = np.linalg.norm(gen_cov - hist_cov, axis=(1, 2))
    cov_fro_scale = np.linalg.norm(hist_cov, axis=(1, 2)) + 1e-12

    by_date = pd.DataFrame(
        {
            "date": dates,
            "mean_mae_vs_realized": np.mean(np.abs(gen_mean - actual), axis=1),
            "mean_mae_vs_historical": np.mean(np.abs(gen_mean - hist_mean), axis=1),
            "vol_mae_vs_realized_abs": np.mean(np.abs(gen_vol - realized_abs), axis=1),
            "vol_mae_vs_historical": np.mean(np.abs(gen_vol - hist_vol), axis=1),
            "cov_fro_error_vs_historical": cov_fro_error,
            "cov_relative_fro_error_vs_historical": cov_fro_error / cov_fro_scale,
            "historical_obs_used": obs_used,
        }
    ).set_index("date")

    by_asset = pd.DataFrame(
        {
            "asset": asset_names,
            "mean_mae_vs_realized": np.mean(np.abs(gen_mean - actual), axis=0),
            "mean_mae_vs_historical": np.mean(np.abs(gen_mean - hist_mean), axis=0),
            "gen_vol_avg": gen_vol.mean(axis=0),
            "realized_abs_return_avg": realized_abs.mean(axis=0),
            "hist_vol_avg": hist_vol.mean(axis=0),
            "vol_mae_vs_realized_abs": np.mean(np.abs(gen_vol - realized_abs), axis=0),
            "vol_mae_vs_historical": np.mean(np.abs(gen_vol - hist_vol), axis=0),
        }
    )

    return by_date, by_asset

