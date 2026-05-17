import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class StockSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len=30):
        """
        df: pre-sorted dataframe by 'ts_code' and 'trade_date'
        """
        self.seq_len = seq_len
        self.features = []
        self.targets = []
        self.dates = []
        self.codes = []
        
        # We must create sequences for each stock independently
        grouped = df.groupby('ts_code')
        for code, group in grouped:
            # We need at least seq_len records
            if len(group) < seq_len:
                continue
                
            group_features = group[feature_cols].values
            group_targets = group[target_col].values
            group_dates = group['trade_date'].astype(str).values
            
            # Create sliding windows
            for i in range(len(group) - seq_len + 1):
                self.features.append(group_features[i : i + seq_len])
                # Target is the label of the LAST day in the sequence
                # (which corresponds to the forward 1-day return after the sequence ends)
                self.targets.append(group_targets[i + seq_len - 1])
                self.dates.append(group_dates[i + seq_len - 1])
                self.codes.append(code)
                
        # Convert to tensors
        self.features = torch.tensor(np.array(self.features), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.dates[idx], self.codes[idx]

def get_dataloaders(parquet_path, seq_len=30, batch_size=512):
    print("Loading parquet data...")
    df = pd.read_parquet(parquet_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # Define features and target
    target_col = 'label_return_1d'
    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in ['ts_code', 'trade_date', target_col]]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Train/Val Split (Time based)
    # Train: 2016-01-01 to 2024-12-31
    # Val: 2025-01-01 to 2026-05-14
    train_df = df[(df['trade_date'] >= '2016-01-01') & (df['trade_date'] <= '2024-12-31')].copy()
    val_df = df[(df['trade_date'] >= '2025-01-01') & (df['trade_date'] <= '2026-05-14')].copy()

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Standardization (Fit on train ONLY to prevent data leakage)
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std() + 1e-8 # prevent division by zero

    train_df.loc[:, feature_cols] = (train_df[feature_cols] - mean) / std
    val_df.loc[:, feature_cols] = (val_df[feature_cols] - mean) / std

    print("Building datasets (sliding windows)... this may take a moment.")
    train_dataset = StockSequenceDataset(train_df, feature_cols, target_col, seq_len=seq_len)
    val_dataset = StockSequenceDataset(val_df, feature_cols, target_col, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(feature_cols)

if __name__ == "__main__":
    # Test
    train_loader, val_loader, num_features = get_dataloaders('../data/processed/csi300_features.parquet', seq_len=30)
    for X, y, dates, codes in train_loader:
        print(f"X batch shape: {X.shape}") # expected: [batch_size, seq_len, num_features]
        print(f"y batch shape: {y.shape}") # expected: [batch_size]
        break
