from __future__ import annotations

import io
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import DDPMConfig


class RollingWindowDataset(Dataset):
    """
    Build (context, target) pairs from a multivariate time series.

    Input series shape:
        [T, D]

    Each sample:
        context: [context_length, D]
        target:  [prediction_length, D]
    """
    def __init__(
        self,
        series: np.ndarray | torch.Tensor,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
        normalize: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        x = torch.as_tensor(series, dtype=torch.float32)
        if x.ndim != 2:
            raise ValueError(f"series must have shape [T, D], got {tuple(x.shape)}")

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.normalize = normalize
        self.eps = eps

        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True).clamp_min(eps)

        if normalize:
            x = (x - self.mean) / self.std

        self.series = x
        self.indices = []
        max_start = len(x) - context_length - prediction_length + 1
        for start in range(0, max_start, stride):
            self.indices.append(start)

        if not self.indices:
            raise ValueError("Not enough data to create even one sample.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
      start = self.indices[idx]
      Lc = self.context_length
      Lp = self.prediction_length

      # context: [Lc, D]
      context = self.series[start : start + Lc]

      # repo-style target:
      # target[j] is the future return after context step j
      targets = []
      for j in range(Lc):
          y0 = start + j + 1
          y1 = y0 + Lp
          targets.append(self.series[y0:y1])  # [Lp, D]

      # target: [Lc, Lp, D]
      target = torch.stack(targets, dim=0)

      return {
          "context": context,
          "target": target,
      }

def build_dataloader(
    series: np.ndarray | torch.Tensor,
    cfg: DDPMConfig,
    shuffle: bool = True,
    stride: int = 1,
    normalize: bool = False,
    num_workers: int = 0,
    drop_last: bool = True,
) -> tuple[RollingWindowDataset, DataLoader]:
    dataset = RollingWindowDataset(
        series=series,
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        stride=stride,
        normalize=normalize,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def load_ff49_monthly_value_weighted_drop_columns(path, start_date="1929-01-01"):
    """
    Load the Fama-French 49 Industry monthly value-weighted returns.

    The function keeps rows from start_date onward, replaces French missing
    values (-99.99 and -999), converts percent returns to decimal returns, and
    drops industries/columns that contain any missing value. Rows are not
    dropped, so the time index remains monthly and continuous for retained
    industries.

    Returns
    -------
    monthly_returns_df : pandas.DataFrame
        Monthly decimal returns with Date index.
    dropped_columns : list[str]
        Industries dropped because they contain at least one missing value.
    missing_count : pandas.Series
        Missing-value count before dropping columns.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if "Average Value Weighted Returns -- Monthly" in line:
            start = i
            break
    if start is None:
        raise ValueError("Could not find Average Value Weighted Returns -- Monthly section.")

    header_idx = start + 1
    end = None
    for j in range(header_idx + 1, len(lines)):
        if lines[j].strip() == "":
            end = j
            break
    if end is None:
        raise ValueError("Could not find end of monthly value-weighted section.")

    block = "".join(lines[header_idx:end])
    df = pd.read_csv(io.StringIO(block))
    df = df.rename(columns={df.columns[0]: "Date"})
    df.columns = [c.strip() for c in df.columns]

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([-99.99, -999.0], np.nan)
    df = df / 100.0
    df = df.loc[start_date:].copy()

    missing_count = df.isna().sum()
    dropped_columns = missing_count[missing_count > 0].index.tolist()
    df_clean = df.drop(columns=dropped_columns)

    if df_clean.isna().sum().sum() != 0:
        raise ValueError("Missing values remain after dropping incomplete columns.")

    return df_clean, dropped_columns, missing_count


def split_train_valid_test(monthly_returns_df, train_end="2010-12-31", valid_start="2011-01-01", valid_end="2015-12-31", test_start="2016-01-01"):
    """Split monthly returns into train, validation target, and test target dataframes."""
    train_df = monthly_returns_df.loc[:train_end].copy()
    valid_target_df = monthly_returns_df.loc[valid_start:valid_end].copy()
    test_target_df = monthly_returns_df.loc[test_start:].copy()
    return train_df, valid_target_df, test_target_df


def standardize_train_valid(train_df, valid_target_df, context_length, eps=1e-8):
    """
    Standardize train and validation data using train statistics only.

    The validation array includes the final context_length observations from
    train, so validation windows have proper historical context.
    """
    train_raw = train_df.values.astype(np.float32)
    valid_with_context_df = pd.concat([train_df.iloc[-context_length:], valid_target_df], axis=0)
    valid_raw = valid_with_context_df.values.astype(np.float32)

    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True) + eps

    train_returns = (train_raw - mean) / std
    valid_returns = (valid_raw - mean) / std
    return train_returns, valid_returns, mean, std, valid_with_context_df
