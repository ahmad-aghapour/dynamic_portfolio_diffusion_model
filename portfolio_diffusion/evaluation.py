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
