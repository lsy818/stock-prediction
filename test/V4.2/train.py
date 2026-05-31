import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats
import collections
from tqdm import tqdm
from dataset import get_dataloaders
from model import EnsembleAttentionGRU

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def calc_ic(preds, targets):
    """
    Calculate Information Coefficient (Pearson Correlation) between predictions and targets
    """
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    if len(preds_np) < 2:
        return 0.0
    ic, _ = scipy.stats.pearsonr(preds_np, targets_np)
    # If standard deviation is 0, pearsonr returns nan
    if np.isnan(ic):
        return 0.0
    return ic

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, target_margin=0.001, max_pairs_per_day=1000):
        super().__init__()
        self.alpha = alpha
        self.target_margin = target_margin
        self.max_pairs_per_day = max_pairs_per_day
        self.mse = nn.MSELoss()

    def forward(self, preds, targets, dates):
        mse_loss = self.mse(preds.squeeze(), targets)
        if self.alpha >= 1.0:
            return mse_loss
            
        preds_sq = preds.squeeze()
        device = preds.device
        
        date_to_indices = collections.defaultdict(list)
        for idx, date in enumerate(dates):
            date_to_indices[date].append(idx)
            
        pairwise_losses = []
        
        for date, indices in date_to_indices.items():
            n = len(indices)
            if n < 2: continue
            
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            p = preds_sq[idx_tensor]
            t = targets[idx_tensor]
            
            p_diff = p.unsqueeze(1) - p.unsqueeze(0)
            t_diff = t.unsqueeze(1) - t.unsqueeze(0)
            
            mask = torch.triu(torch.abs(t_diff) > self.target_margin, diagonal=1)
            
            if not mask.any():
                continue
                
            valid_p_diff = p_diff[mask]
            valid_t_diff = t_diff[mask]
            
            y_sign = torch.sign(valid_t_diff)
            
            if len(valid_p_diff) > self.max_pairs_per_day:
                perm = torch.randperm(len(valid_p_diff), device=device)[:self.max_pairs_per_day]
                valid_p_diff = valid_p_diff[perm]
                y_sign = y_sign[perm]
                
            loss = torch.nn.functional.softplus(-y_sign * valid_p_diff).mean()
            pairwise_losses.append(loss)
            
        if len(pairwise_losses) == 0:
            return mse_loss
            
        pairwise_loss = torch.stack(pairwise_losses).mean()
        return self.alpha * mse_loss + (1.0 - self.alpha) * pairwise_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for X, y, dates, _ in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        preds = model(X)
        if isinstance(criterion, HybridLoss):
            loss = criterion(preds, y, dates)
        else:
            loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y, dates, _ in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            if isinstance(criterion, HybridLoss):
                loss = criterion(preds, y, dates)
            else:
                loss = criterion(preds, y)
            
            total_loss += loss.item()
            all_preds.append(preds)
            all_targets.append(y)
            
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    ic = calc_ic(all_preds, all_targets)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, ic

def train_model(data_path, train_period, val_period, save_path, seq_len=15, epochs=15, batch_size=12288, lr=1e-3, patience=5, target_col='label_return_1d', loss_type='mse', alpha=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Preparing Dataloaders for Train: {train_period}, Val: {val_period} with target: {target_col}...")
    train_loader, val_loader, _, num_features = get_dataloaders(
        data_path, seq_len=seq_len, batch_size=batch_size,
        train_period=train_period, val_period=val_period, test_period=None,
        target_col=target_col
    )
    
    print("Initializing Model...")
    model = EnsembleAttentionGRU(
        input_size=num_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        num_models=3
    ).to(device)
    
    if loss_type == 'hybrid':
        print(f"Using HybridLoss with alpha={alpha}")
        criterion = HybridLoss(alpha=alpha)
    else:
        print("Using MSELoss")
        criterion = nn.MSELoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_ic = -float('inf')
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    print("Starting Training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_ic = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val IC: {val_ic:.4f}")
        
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Best Model Saved to {save_path}!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    return model

def main():
    set_seed(42)
    data_path = '../../data/processed/800_stocks_features.parquet'
    train_model(
        data_path=data_path,
        train_period=('2016-01-01', '2024-12-31'),
        val_period=('2025-01-01', '2025-12-31'),
        save_path='checkpoints/best_ensemble.pth',
        target_col='label_return_5d',
        loss_type='hybrid',
        alpha=0.5
    )

if __name__ == '__main__':
    main()
