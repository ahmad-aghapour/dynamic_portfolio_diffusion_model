
import math
import time
import copy
import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
    LambdaLR,
)
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# EMA
# ============================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self):
        for p, s in zip(self.params, self.shadow):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.copy_(s)

    @torch.no_grad()
    def restore(self):
        for p, b in zip(self.params, self.backup):
            p.copy_(b)
        self.backup = None


# ============================================================
# Config
# ============================================================

@dataclasses.dataclass
class TrainCfg:
    optim_name: str = "AdamW"
    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.9

    scheduler: Optional[str] = "cosine"
    num_epochs: int = 200

    use_ema: bool = True
    ema_decay: float = 0.999

    max_grad_norm: Optional[float] = 1.0
    amp: bool = True
    patience: int = 40

    seed: int = 42
    save_path: str = "best_ddpm.pt"

    # Important for your current model because len_proj is created lazily
    force_lazy_modules: bool = True


# ============================================================
# Helpers
# ============================================================

def unpack_batch(batch):
    """
    Works with current RollingWindowDataset:
        batch["context"], batch["target"]

    Also still supports old tuple datasets:
        batch_x, batch_y
    """
    if isinstance(batch, dict):
        context = batch["context"]
        target = batch["target"]
    else:
        context, target = batch

    return context, target


def build_optimizer(model: nn.Module, cfg: TrainCfg):
    kwargs = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
    }

    name = cfg.optim_name.lower()

    if name in {"adam", "adamw", "adamax", "nadam", "radam"}:
        kwargs["betas"] = cfg.betas

    if name in {"sgd", "rmsprop"}:
        kwargs["momentum"] = cfg.momentum

    optim_cls = getattr(optim, cfg.optim_name)
    return optim_cls(model.parameters(), **kwargs)


def build_scheduler(optimizer, cfg: TrainCfg, steps_per_epoch: int):
    if cfg.scheduler is None or cfg.scheduler.lower() == "none":
        return None

    name = cfg.scheduler.lower()

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epochs,
            eta_min=cfg.lr * 0.05,
        )

    if name == "cosine_restart":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=cfg.lr * 0.05,
        )

    if name == "step":
        return StepLR(
            optimizer,
            step_size=30,
            gamma=0.5,
        )

    if name == "exp":
        return ExponentialLR(
            optimizer,
            gamma=0.97,
        )

    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        )

    if name == "1cycle":
        return OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=cfg.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.10,
            anneal_strategy="cos",
        )

    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


@torch.no_grad()
def force_create_lazy_modules(model, train_loader, device):
    """
    Your current UNet creates len_proj inside the first forward pass.
    If optimizer is built before that, len_proj parameters are not optimized.

    This dummy pass forces those modules to exist before optimizer creation.
    """
    model.to(device)
    model.eval()

    batch = next(iter(train_loader))
    context, target = unpack_batch(batch)

    context = context[:2].to(device)
    target = target[:2].to(device)

    _loss, _info = model.training_loss(context, target)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader, device, ema: Optional[EMA] = None):
    if loader is None:
        return None

    if ema is not None:
        ema.apply_shadow()

    model.eval()

    loss_sum = 0.0
    n_obs = 0

    for batch in loader:
        context, target = unpack_batch(batch)

        context = context.to(device)
        target = target.to(device)

        loss, _ = model.training_loss(context, target)

        bs = context.size(0)
        loss_sum += loss.item() * bs
        n_obs += bs

    if ema is not None:
        ema.restore()

    return loss_sum / max(n_obs, 1)


# ============================================================
# Main fit function for current DDPM
# ============================================================

def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainCfg,
):
    seed_everything(cfg.seed)

    device = torch.device(device)
    model.to(device)

    # Critical for your current model architecture
    if cfg.force_lazy_modules:
        force_create_lazy_modules(model, train_loader, device)

    optimizer = build_optimizer(model, cfg)

    scheduler = build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        steps_per_epoch=len(train_loader),
    )

    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None

    step_per_batch = isinstance(scheduler, (OneCycleLR, LambdaLR))

    best_loss = math.inf
    best_model = None
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    start = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()

        train_loss_sum = 0.0
        n_obs = 0

        prog = tqdm(
            train_loader,
            leave=False,
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
        )

        for batch in prog:
            context, target = unpack_batch(batch)

            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                loss, _ = model.training_loss(context, target)

            scaler.scale(loss).backward()

            if cfg.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update()

            # Only step per batch for OneCycle/Lambda schedules
            if scheduler is not None and step_per_batch:
                scheduler.step()

            bs = context.size(0)
            train_loss_sum += loss.item() * bs
            n_obs += bs

            prog.set_postfix(
                loss=f"{loss.item():.5f}",
                lr=f"{get_lr(optimizer):.2e}",
            )

        train_loss = train_loss_sum / max(n_obs, 1)

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            ema=ema,
        )

        metric_loss = val_loss if val_loss is not None else train_loss

        # Epoch-level schedulers step here, not inside batch loop
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metric_loss)
            elif not step_per_batch:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(get_lr(optimizer))

        elapsed = (time.time() - start) / 60.0

        if val_loss is None:
            print(
                f"[{epoch + 1:03d}/{cfg.num_epochs:03d}] "
                f"train={train_loss:.6f} "
                f"lr={get_lr(optimizer):.2e} "
                f"t={elapsed:.1f}m"
            )
        else:
            print(
                f"[{epoch + 1:03d}/{cfg.num_epochs:03d}] "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"lr={get_lr(optimizer):.2e} "
                f"t={elapsed:.1f}m"
            )

        if metric_loss < best_loss - 1e-6:
            best_loss = metric_loss
            epochs_no_improve = 0

            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), cfg.save_path)
                best_model = copy.deepcopy(model)
                ema.restore()
            else:
                torch.save(model.state_dict(), cfg.save_path)
                best_model = copy.deepcopy(model)

        else:
            epochs_no_improve += 1

            if epochs_no_improve >= cfg.patience:
                print("Early stopping")
                break

    if best_model is None:
        best_model = copy.deepcopy(model)

    return best_model, history