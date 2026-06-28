from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DDPMConfig


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)

        exponent = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half_dim - 1, 1))
        )
        emb = timesteps.float().unsqueeze(1) * exponent.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class FiLM(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_scale_shift(cond_vec).chunk(2, dim=-1)
        return (1.0 + gamma.unsqueeze(-1)) * x + beta.unsqueeze(-1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.film1 = FiLM(out_channels, cond_dim)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.film2 = FiLM(out_channels, cond_dim)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.film1(x, cond_vec)
        x = F.silu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film2(x, cond_vec)
        x = F.silu(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels, cond_dim)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        return self.block(x, cond_vec)

class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels, cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond_vec: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([skip, x], dim=1)
        return self.block(x, cond_vec)



class ConditionalMLPDenoiser(nn.Module):
    def __init__(self, asset_dim, cond_dim, time_embed_dim=128, hidden_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(asset_dim + cond_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, asset_dim),
        )

    def forward(self, x, t, cond):
        x_vec = x[:, 0, :]
        cond_vec = cond[:, 0, :]
        time_vec = self.time_mlp(t)

        h = torch.cat([x_vec, cond_vec, time_vec], dim=-1)
        eps = self.net(h)

        return eps.unsqueeze(1)
class UNet1DSameRes(nn.Module):
    """
    Same-resolution 1D U-Net.

    Keeps the repo's basic idea:
      - x shape is [B, prediction_length, target_dim]
      - cond shape is [B, 1, hidden_dim]
      - if hidden_dim != target_dim, project condition length to target_dim
    """
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        time_embed_dim: int = 128,
        base_channels: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.cond_channels = cond_channels
        self.len_proj = nn.ModuleDict()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        total_in = in_channels + cond_channels
        self.inc = ConvBlock(total_in, base_channels, time_embed_dim)

        ch = base_channels
        enc_channels = [ch]
        self.downs = nn.ModuleList()
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2, time_embed_dim))
            ch *= 2
            enc_channels.append(ch)

        self.bottleneck = ConvBlock(ch, ch, time_embed_dim)

        self.ups = nn.ModuleList()
        for skip_ch in reversed(enc_channels[:-1]):
            self.ups.append(Up(ch, skip_ch, skip_ch, time_embed_dim))
            ch = skip_ch

        self.out = nn.Conv1d(base_channels, in_channels, kernel_size=1)

    def _project_cond_length(self, cond: torch.Tensor, target_len: int) -> torch.Tensor:
        b, c, cond_len = cond.shape
        key = f"{cond_len}->{target_len}"
        if key not in self.len_proj:
            self.len_proj[key] = nn.Linear(cond_len, target_len, bias=False).to(cond.device)
        proj = self.len_proj[key]
        cond = proj(cond.reshape(b * c, cond_len)).reshape(b, c, target_len)
        return cond

    def forward(
        self,
        x: torch.Tensor,              # [B, pred_len, target_dim]
        t: torch.Tensor,              # [B]
        cond: Optional[torch.Tensor], # [B, 1, hidden_dim]
    ) -> torch.Tensor:
        if self.cond_channels > 0:
            if cond is None:
                raise ValueError("cond is required")
            if cond.shape[1] != self.cond_channels:
                raise ValueError(
                    f"Expected cond_channels={self.cond_channels}, got {cond.shape[1]}"
                )
            if cond.shape[-1] != x.shape[-1]:
                cond = self._project_cond_length(cond, x.shape[-1])
            x = x + cond
            x = torch.cat([x, cond], dim=1)

        cond_vec = self.time_mlp(t)

        skips = []
        x = self.inc(x, cond_vec)
        skips.append(x)

        for down in self.downs:
            x = down(x, cond_vec)
            skips.append(x)

        x = self.bottleneck(x, cond_vec)

        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip, cond_vec)

        return self.out(x)

class ConditionalRecurrentDDPM(nn.Module):
    """
    Recurrent conditioner + 1D U-Net denoiser.

    Training:
        - sample diffusion step t
        - add noise to target x0 -> xt
        - predict epsilon
        - optimize MSE(eps_pred, eps)

    Sampling:
        - DDIM update with configurable eta
        - use eta=1.0 to match your requested setting
    """
    def __init__(self, cfg: DDPMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.target_dim = cfg.input_dim
        self.prediction_length = cfg.prediction_length
        self.feature_dim = cfg.rnn_hidden_dim
        self.scale_condition = cfg.scale_condition

        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}[cfg.rnn_type.upper()]
        self.rnn = rnn_cls(
            input_size=cfg.input_dim,
            hidden_size=cfg.rnn_hidden_dim,
            num_layers=cfg.rnn_layers,
            dropout=cfg.rnn_dropout if cfg.rnn_layers > 1 else 0.0,
            batch_first=True,
        )

        if cfg.denoiser_type.lower() == "mlp":
            self.denoiser = ConditionalMLPDenoiser(
                asset_dim=cfg.input_dim,
                cond_dim=cfg.rnn_hidden_dim,
                time_embed_dim=cfg.time_embed_dim,
                hidden_dim=256,
            )
        elif cfg.denoiser_type.lower() == "unet":
                self.denoiser = UNet1DSameRes(
                    in_channels=cfg.prediction_length,
                    cond_channels=1,
                    time_embed_dim=cfg.time_embed_dim,
                    base_channels=cfg.unet_base_channels,
                    depth=cfg.unet_depth,
                )
        else:
                raise ValueError(f"Unknown denoiser_type: {cfg.denoiser_type}")

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)

        self.num_steps = cfg.diffusion_steps

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer(
            "sqrt_recipm1_alpha_bars",
            torch.sqrt(torch.clamp(1.0 / alpha_bars - 1.0, min=0.0)),
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: [B, context_length, input_dim]
        returns cond: [B, 1, hidden_dim]
        """
        _, hidden = self.rnn(context)

        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]

        cond = hidden[-1].unsqueeze(1)  # [B, 1, H]
        cond = cond * self.scale_condition
        return cond
    def encode_context_all(self, context: torch.Tensor) -> torch.Tensor:
      """
      context: [B, Lc, D]

      returns:
          cond_seq: [B, Lc, H]

      This gives one conditioning vector for every RNN time step.
      """
      rnn_outputs, _ = self.rnn(context)  # [B, Lc, H]
      return rnn_outputs * self.scale_condition
    def predict_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(x_t, t, cond)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def training_loss(
      self,
      context: torch.Tensor,
      target: torch.Tensor,
  ) -> Tuple[torch.Tensor, Dict[str, float]]:
      """
      Supports two formats:

      Old format:
          context: [B, Lc, D]
          target:  [B, Lp, D]

      Repo-style format:
          context: [B, Lc, D]
          target:  [B, Lc, Lp, D]
      """

      # ---------------------------------------------------------
      # Old single-target training
      # ---------------------------------------------------------
      if target.ndim == 3:
          b = context.size(0)

          t = torch.randint(
              0,
              self.num_steps,
              size=(b,),
              device=context.device,
              dtype=torch.long,
          )

          noise = torch.randn_like(target)
          x_t = self.q_sample(target, t, noise)

          cond = self.encode_context(context)  # [B, 1, H]
          eps_pred = self.predict_eps(x_t, t, cond)

          loss = F.mse_loss(eps_pred, noise)

          return loss, {
              "loss": float(loss.detach().item()),
              "mode": "single_target",
          }

      # ---------------------------------------------------------
      # Repo-style seq2seq training
      # ---------------------------------------------------------
      if target.ndim != 4:
          raise ValueError(
              f"target must have shape [B, Lp, D] or [B, Lc, Lp, D], "
              f"got {tuple(target.shape)}"
          )

      B, Lc, Lp, D = target.shape

      if Lc != context.size(1):
          raise ValueError(
              f"target Lc={Lc} must equal context length={context.size(1)}"
          )

      # RNN output for every time step.
      cond_seq = self.encode_context_all(context)  # [B, Lc, H]

      # Flatten B and Lc, same idea as their repo.
      target_flat = target.reshape(B * Lc, Lp, D)      # [B*Lc, Lp, D]
      cond_flat = cond_seq.reshape(B * Lc, 1, -1)      # [B*Lc, 1, H]

      t = torch.randint(
          0,
          self.num_steps,
          size=(B * Lc,),
          device=context.device,
          dtype=torch.long,
      )

      noise = torch.randn_like(target_flat)
      x_t = self.q_sample(target_flat, t, noise)

      eps_pred = self.predict_eps(x_t, t, cond_flat)

      seq_loss  = F.mse_loss(eps_pred, noise)
      target_last = target[:, -1, :, :]          # [B, Lp, D]
      cond_last = cond_seq[:, -1, :].unsqueeze(1)

      t_last = torch.randint(
            0,
            self.num_steps,
            size=(B,),
            device=context.device,
            dtype=torch.long,
        )

      noise_last = torch.randn_like(target_last)
      x_t_last = self.q_sample(target_last, t_last, noise_last)
      eps_pred_last = self.predict_eps(x_t_last, t_last, cond_last)

      last_loss = F.mse_loss(eps_pred_last, noise_last)  
      loss = (
            self.cfg.seq_loss_weight * seq_loss
            + self.cfg.last_loss_weight * last_loss
        )

      return loss, {
            "loss": float(loss.detach().item()),
            "seq_loss": float(seq_loss.detach().item()),
            "last_loss": float(last_loss.detach().item()),
            "mode": "seq2seq_weighted",
            "effective_targets": B * Lc,
        }
    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss, _ = self.training_loss(context, target)
        return loss

    @torch.no_grad()
    def ddim_sample(
        self,
        context: torch.Tensor,
        num_samples: int = 1,
        eta: float = 1.0,
        clip_x0: Optional[float] = None,
    ) -> torch.Tensor:
        """
        context:
            [B, context_length, D]

        returns:
            [B, num_samples, prediction_length, D]
        """
        self.eval()
        device = context.device
        batch_size = context.size(0)

        cond = self.encode_context(context)                      # [B, 1, H]
        cond = cond.repeat_interleave(num_samples, dim=0)       # [B*num_samples, 1, H]

        x = torch.randn(
            batch_size * num_samples,
            self.prediction_length,
            self.target_dim,
            device=device,
        )

        for step in reversed(range(self.num_steps)):
            t = torch.full((x.size(0),), step, device=device, dtype=torch.long)
            eps = self.predict_eps(x, t, cond)

            alpha_bar_t = self.alpha_bars[step]
            alpha_bar_prev = self.alpha_bars_prev[step]

            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

            x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

            if clip_x0 is not None:
                x0_pred = x0_pred.clamp(-clip_x0, clip_x0)

            if step == 0:
                x = x0_pred
                continue

            sigma_t = eta * torch.sqrt(
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                * (1.0 - alpha_bar_t / alpha_bar_prev)
            )
            noise = torch.randn_like(x)
            coeff_eps = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=0.0))

            x = (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + coeff_eps * eps
                + sigma_t * noise
            )

        x = x.view(batch_size, num_samples, self.prediction_length, self.target_dim)
        return x

    @torch.no_grad()
    def sample_autoregressive(
        self,
        history: torch.Tensor,
        horizon: int,
        num_samples: int = 1,
        eta: float = 1.0,
        clip_x0: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build (context, target) pairs from a multivariate return series.

        Input series shape:
            [T, D]

        Each sample:
            context: [context_length, D]      historical returns
            target:  [prediction_length, D]   future returns
        """
        self.eval()
        device = next(self.parameters()).device

        if history.ndim != 2:
            raise ValueError(f"history must have shape [context_length, D], got {tuple(history.shape)}")
        if history.shape[0] != self.cfg.context_length:
            raise ValueError(
                f"history length must equal context_length={self.cfg.context_length}, "
                f"got {history.shape[0]}"
            )

        context = history.unsqueeze(0).repeat(num_samples, 1, 1).to(device)
        all_returns = []
        all_features = []

        for _ in range(horizon):
            cond = self.encode_context(context)                     # [num_samples, 1, H]
            all_features.append(cond[:, 0, :].detach().cpu())

            block = self.ddim_sample(
                context=context,
                num_samples=1,
                eta=eta,
                clip_x0=clip_x0,
            )[:, 0]                                                # [num_samples, pred_len, D]

            next_return = block[:, 0, :]                           # [num_samples, D]
            all_returns.append(next_return.detach().cpu())

            context = torch.cat([context[:, 1:, :], next_return.unsqueeze(1)], dim=1)

        sampled_returns = torch.stack(all_returns, dim=0)
        sampled_features = torch.stack(all_features, dim=0)
        return sampled_returns, sampled_features
