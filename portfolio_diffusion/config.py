from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class DDPMConfig:
    """Configuration for the conditional recurrent DDPM."""

    input_dim: int
    context_length: int = 12
    prediction_length: int = 1

    rnn_type: str = "LSTM"  # "GRU" or "LSTM"
    rnn_hidden_dim: int = 128
    rnn_layers: int = 2
    rnn_dropout: float = 0.05

    unet_base_channels: int = 64
    unet_depth: int = 3
    time_embed_dim: int = 128

    diffusion_steps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    lr: float = 2e-4
    batch_size: int = 16
    epochs: int = 30
    grad_clip: float = 1.0
    weight_decay: float = 1e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    scale_condition: float = 1.0
