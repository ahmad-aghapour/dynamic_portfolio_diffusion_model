from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .models import ConditionalRecurrentDDPM


def generate_scenarios(
    model: ConditionalRecurrentDDPM,
    return_history_window: np.ndarray | torch.Tensor,
    horizon: int,
    num_samples: int,
    eta: float = 1.0,
    clip_x0: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate return scenarios only.

    Input:
        return_history_window: [context_length, D]
            Historical returns, not prices.

    Returns:
        {
            "returns":  [horizon, num_samples, D],
            "features": [horizon, num_samples, H],
        }
    """
    device = next(model.parameters()).device

    history = torch.as_tensor(
        return_history_window,
        dtype=torch.float32,
        device=device,
    )

    sampled_returns, sampled_features = model.sample_autoregressive(
        history=history,
        horizon=horizon,
        num_samples=num_samples,
        eta=eta,
        clip_x0=clip_x0,
    )

    return {
        "returns": sampled_returns,
        "features": sampled_features,
    }

@torch.no_grad()
def generate_next_month_scenarios(
    model,
    history_window_standardized,
    num_scenarios=500,
    eta=1.0,
    clip_x0=None,
):
    """
    Generate next-month return scenarios from one standardized history window.

    Parameters
    ----------
    history_window_standardized : array-like, shape [context_length, num_assets]

    Returns
    -------
    torch.Tensor, shape [num_scenarios, num_assets]
        Standardized generated scenarios.
    """
    model.eval()
    device = next(model.parameters()).device

    history_tensor = torch.as_tensor(
        history_window_standardized,
        dtype=torch.float32,
        device=device,
    )

    samples, _ = model.sample_autoregressive(
        history=history_tensor,
        horizon=1,
        num_samples=num_scenarios,
        eta=eta,
        clip_x0=clip_x0,
    )

    return samples[0].detach().cpu()
