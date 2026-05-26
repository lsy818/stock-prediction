import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import collections

class DateBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, dates_per_batch=20, shuffle=True):
        self.dates_per_batch = dates_per_batch
        self.shuffle = shuffle
        
        # Group dataset indices by date
        self.date_to_indices = collections.defaultdict(list)
        for i, idx in enumerate(dataset.valid_indices):
            self.date_to_indices[dataset.dates[idx]].append(i)
            
        self.dates = sorted(self.date_to_indices.keys())
        self.num_batches = len(self.dates) // dates_per_batch
            
    def __iter__(self):
        dates = self.dates.copy()
        if self.shuffle:
            np.random.shuffle(dates)
            
        num_full_batches = len(dates) // self.dates_per_batch
        remainder = len(dates) % self.dates_per_batch

        for i in range(num_full_batches):
            batch_dates = dates[i * self.dates_per_batch : (i + 1) * self.dates_per_batch]
            batch_indices = []
            for d in batch_dates:
                batch_indices.extend(self.date_to_indices[d])
            yield batch_indices

        if remainder > 0 and not self.shuffle:  # 验证/测试集保留余数
            batch_dates = dates[num_full_batches * self.dates_per_batch:]
            batch_indices = []
            for d in batch_dates:
                batch_indices.extend(self.date_to_indices[d])
            yield batch_indices
            
    def __len__(self):
        return self.num_batches

class StockSequenceDataset(Dataset):
    def __init__(self, features, targets, codes, dates, stock_changes, seq_len=15, target_date_start=None, target_date_end=None):
        self.seq_len = seq_len
        self.features = features
        self.targets = targets
        self.codes = codes
        self.dates = dates
        
        self.valid_indices = []
        
        # Fast extraction of valid indices
        current_stock_start = 0
        for i in range(len(features)):
            if stock_changes[i]:
                current_stock_start = i
                
            if i - current_stock_start + 1 >= seq_len:
                target_date = self.dates[i]
                if target_date_start is not None and target_date < target_date_start:
                    continue
                if target_date_end is not None and target_date > target_date_end:
                    continue
                self.valid_indices.append(i)
                
        self.valid_indices = np.array(self.valid_indices, dtype=np.int32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len + 1
        
        X = self.features[start_idx : end_idx + 1]
        y = self.targets[end_idx]
        date = self.dates[end_idx]
        code = self.codes[end_idx]
        
        return X, y, date, code

def get_dataloaders(parquet_path, seq_len=15, 
                    train_period=('2016-01-01', '2024-12-31'),
                    val_period=('2025-01-01', '2025-12-31'),
                    test_period=('2026-01-01', '2026-12-31'),
                    target_col='label_return_1d'):
    print("Loading parquet data...")
    df = pd.read_parquet(parquet_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # --- Board-Specific Time Filtering (Regime Shift Alignment) ---
    # 创业板 (300) 2020年8月24日引入20%涨跌停，舍弃此前数据
    mask_chinext = df['ts_code'].str.startswith('300')
    df_chinext = df[mask_chinext & (df['trade_date'] >= '2020-08-24')]
    
    # 其他板块（主板始终10%，科创板天生20%），放开历史至2016年
    mask_others = ~mask_chinext
    df_others = df[mask_others & (df['trade_date'] >= '2016-01-01')]
    
    df = pd.concat([df_chinext, df_others], ignore_index=True)
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    # --------------------------------------------------------------

    # Define features and target
    # target_col is passed as an argument
    # Strictly exclude ALL label columns to prevent look-ahead leakage
    label_cols = ['label_return_1d', 'label_return_3d', 'label_return_5d']
    feature_cols = [c for c in df.columns if c not in ['ts_code', 'trade_date'] + label_cols]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Extract feature columns as a float32 numpy array immediately to save memory
    features_np = df[feature_cols].values.astype(np.float32)
    targets_np = df[target_col].values.astype(np.float32)
    
    # Clean up dataframe's feature columns to free up memory
    df.drop(columns=feature_cols + [target_col], inplace=True)
    import gc; gc.collect()

    # Train/Val/Test Split (Time based filtering on targets)
    train_start, train_end = pd.to_datetime(train_period[0]), pd.to_datetime(train_period[1])
    val_start, val_end = (pd.to_datetime(val_period[0]), pd.to_datetime(val_period[1])) if val_period is not None else (None, None)
    test_start, test_end = (pd.to_datetime(test_period[0]), pd.to_datetime(test_period[1])) if test_period is not None else (None, None)

    # Standardization (Fit on train ONLY to prevent data leakage)
    train_mask = ((df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)).values
    
    train_features = features_np[train_mask]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0) + 1e-8
    del train_features # Free memory
    
    # Apply standardization globally IN-PLACE
    features_np -= mean
    features_np /= std

    # Pre-create tensors and arrays once to save RAM
    # from_numpy shares memory with the numpy array, 0 byte copy!
    features_tensor = torch.from_numpy(features_np)
    targets_tensor = torch.from_numpy(targets_np)
    codes_array = df['ts_code'].values
    dates_array = df['trade_date'].dt.strftime('%Y-%m-%d').values
    stock_changes = (df['ts_code'] != df['ts_code'].shift(1)).values

    print("Building datasets (sliding windows)... this may take a moment.")
    train_start_str = train_start.strftime('%Y-%m-%d')
    train_end_str = train_end.strftime('%Y-%m-%d')
    
    train_dataset = StockSequenceDataset(features_tensor, targets_tensor, codes_array, dates_array, stock_changes, seq_len=seq_len, 
                                         target_date_start=train_start_str, target_date_end=train_end_str)
    train_sampler = DateBatchSampler(train_dataset, dates_per_batch=20, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True, num_workers=4)

    val_dataset = None
    val_loader = None
    if val_period is not None:
        val_start_str = val_start.strftime('%Y-%m-%d')
        val_end_str = val_end.strftime('%Y-%m-%d')
        val_dataset = StockSequenceDataset(features_tensor, targets_tensor, codes_array, dates_array, stock_changes, seq_len=seq_len,
                                           target_date_start=val_start_str, target_date_end=val_end_str)
        val_sampler = DateBatchSampler(val_dataset, dates_per_batch=20, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, pin_memory=True, num_workers=4)

    test_dataset = None
    test_loader = None
    if test_period is not None:
        test_start_str = test_start.strftime('%Y-%m-%d')
        test_end_str = test_end.strftime('%Y-%m-%d')
        test_dataset = StockSequenceDataset(features_tensor, targets_tensor, codes_array, dates_array, stock_changes, seq_len=seq_len,
                                            target_date_start=test_start_str, target_date_end=test_end_str)
        test_sampler = DateBatchSampler(test_dataset, dates_per_batch=20, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, pin_memory=True, num_workers=4)

    val_len = len(val_dataset) if val_dataset is not None else 0
    test_len = len(test_dataset) if test_dataset is not None else 0
    print(f"Train targets: {len(train_dataset)}, Val targets: {val_len}, Test targets: {test_len}")

    print(f"Target stats: mean={targets_np[train_mask].mean():.4f}, std={targets_np[train_mask].std():.4f}")

    return train_loader, val_loader, test_loader, len(feature_cols)

if __name__ == "__main__":
    # Test
    train_loader, val_loader, num_features = get_dataloaders('../data/processed/csi300_features.parquet', seq_len=30)
    for X, y, dates, codes in train_loader:
        print(f"X batch shape: {X.shape}") # expected: [batch_size, seq_len, num_features]
        print(f"y batch shape: {y.shape}") # expected: [batch_size]
        break
