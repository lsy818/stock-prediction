import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class StockSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len=30, target_date_start=None, target_date_end=None):
        """
        df: pre-sorted dataframe by 'ts_code' and 'trade_date'
        """
        self.seq_len = seq_len
        self.features = []
        self.targets = []
        self.dates = []
        self.codes = []
        
        # Ensure target dates are strings for comparison with group_dates (which were cast to str)
        if target_date_start is not None:
            target_date_start = pd.to_datetime(target_date_start).strftime('%Y-%m-%d')
        if target_date_end is not None:
            target_date_end = pd.to_datetime(target_date_end).strftime('%Y-%m-%d')
        
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
                target_date = group_dates[i + seq_len - 1]
                
                # Filter by actual target date to ensure strict no-overlap
                if target_date_start is not None and target_date < target_date_start:
                    continue
                if target_date_end is not None and target_date > target_date_end:
                    continue
                    
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

    # Train/Val/Test Split (Time based filtering on targets)
    train_start, train_end = pd.to_datetime('2016-01-04'), pd.to_datetime('2024-12-31')
    val_start, val_end = pd.to_datetime('2025-01-01'), pd.to_datetime('2025-12-31')
    test_start, test_end = pd.to_datetime('2026-01-01'), pd.to_datetime('2026-12-31')

    # Standardization (Fit on train ONLY to prevent data leakage)
    train_mask = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    mean = df[train_mask][feature_cols].mean()
    std = df[train_mask][feature_cols].std() + 1e-8 # prevent division by zero

    # Apply standardization globally
    df.loc[:, feature_cols] = (df[feature_cols] - mean) / std

    print("Building datasets (sliding windows)... this may take a moment.")
    train_dataset = StockSequenceDataset(df, feature_cols, target_col, seq_len=seq_len, 
                                         target_date_start=train_start, target_date_end=train_end)
    val_dataset = StockSequenceDataset(df, feature_cols, target_col, seq_len=seq_len,
                                       target_date_start=val_start, target_date_end=val_end)
    test_dataset = StockSequenceDataset(df, feature_cols, target_col, seq_len=seq_len,
                                        target_date_start=test_start, target_date_end=test_end)

    print(f"Train targets: {len(train_dataset)}, Val targets: {len(val_dataset)}, Test targets: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, len(feature_cols)

if __name__ == "__main__":
    # Test
    train_loader, val_loader, num_features = get_dataloaders('../data/processed/csi300_features.parquet', seq_len=30)
    for X, y, dates, codes in train_loader:
        print(f"X batch shape: {X.shape}") # expected: [batch_size, seq_len, num_features]
        print(f"y batch shape: {y.shape}") # expected: [batch_size]
        break
