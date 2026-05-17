"""
Sliding window dataset using numpy arrays for fast indexing.
Performs per-sample normalization to avoid data leakage.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import SEQ_LEN, TARGET_N, BATCH_SIZE, NUM_WORKERS


class StockDataset(Dataset):
    """PyTorch Dataset backed by numpy arrays for speed."""

    def __init__(self, df, feature_cols, seq_len=SEQ_LEN, target_n=TARGET_N,
                 target_col="_target_cs", sample_step=1):
        self.seq_len = seq_len
        self.sample_step = sample_step

        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        self.feature_cols = [c for c in feature_cols if c in df.columns]

        # Ensure target columns exist
        if "_target_raw" not in df.columns:
            df["_target_raw"] = df.groupby("ts_code")["close"].transform(
                lambda x: x.shift(-target_n) / x - 1.0
            )
        if "_target_cs" not in df.columns and target_col == "_target_cs":
            df["_target_cs"] = df.groupby("trade_date")["_target_raw"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )

        self.target_col = target_col if target_col in df.columns else "_target_raw"
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)

        # Store metadata for mapping predictions back
        self.codes = df["ts_code"].values
        self.dates = df["trade_date"].values

        # Convert to numpy arrays for fast access
        self.features = df[self.feature_cols].values.astype(np.float32)
        self.targets = df[self.target_col].values.astype(np.float32)

        # Build stock boundaries: [(start_idx, end_idx), ...] per stock
        self.stock_boundaries = []
        grouped = df.groupby("ts_code")
        for code, group in grouped:
            if len(group) > self.seq_len:
                idx = group.index.values  # numpy array of integer positions
                self.stock_boundaries.append((idx[0], idx[-1] + 1))

        # Build sample index
        self.samples = self._build_sample_index()

    def _build_sample_index(self):
        samples = []
        for start, end in self.stock_boundaries:
            stock_len = end - start
            for i in range(self.seq_len, stock_len):
                samples.append(start + i)
        # Subsample
        if self.sample_step > 1:
            samples = samples[::self.sample_step]
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        end_pos = self.samples[idx]          # absolute position in the array
        start_pos = end_pos - self.seq_len

        # Fast numpy slicing
        window = self.features[start_pos:end_pos]  # [seq_len, F]
        target = self.targets[end_pos]              # scalar

        # Per-sample normalization
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-8
        window = (window - mean) / std

        return (
            torch.from_numpy(window.copy()),
            torch.tensor(float(target), dtype=torch.float32),
        )


def collate_fn(batch):
    X = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch]).unsqueeze(-1)
    return X, y


def create_dataloaders(
    df_train, df_val, feature_cols,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    sample_step=5,
):
    train_dataset = StockDataset(df_train, feature_cols, sample_step=sample_step)
    val_dataset = StockDataset(df_val, feature_cols, sample_step=1)

    print(f"[INFO] Train samples: {len(train_dataset)} (step={sample_step}), "
          f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=False,
    )
    return train_loader, val_loader


def compute_cross_sectional_targets(df, target_n=TARGET_N):
    """Compute raw and cross-sectionally standardized forward returns."""
    df = df.copy()

    if "_target_raw" not in df.columns:
        df["_target_raw"] = df.groupby("ts_code")["close"].transform(
            lambda x: x.shift(-target_n) / x - 1.0
        )

    df["_target_cs"] = df.groupby("trade_date")["_target_raw"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )

    q_low = df["_target_cs"].quantile(0.01)
    q_high = df["_target_cs"].quantile(0.99)
    df["_target_cs"] = df["_target_cs"].clip(q_low, q_high)
    return df
