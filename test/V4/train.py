import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats
from tqdm import tqdm
from dataset import get_dataloaders
from model import EnsembleAttentionGRU

# 用显存换取更快的训练速度
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

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
    def __init__(self, alpha=0.5, target_margin=0.5, max_pairs_per_day=200):
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
        n = len(dates)

        # dates 已经是 int32，直接用 tensor 向量化分组
        date_tensor = torch.tensor(dates, dtype=torch.long, device=device)
        unique_dates, inverse = torch.unique(date_tensor, return_inverse=True)

        pairwise_losses = []

        for d_idx in range(len(unique_dates)):
            mask = (inverse == d_idx)
            indices = torch.where(mask)[0]
            n_d = len(indices)
            if n_d < 2:
                continue

            p = preds_sq[indices]
            t = targets[indices]

            p_diff = p.unsqueeze(1) - p.unsqueeze(0)
            t_diff = t.unsqueeze(1) - t.unsqueeze(0)

            valid_mask = torch.triu(torch.abs(t_diff) > self.target_margin, diagonal=1)
            if not valid_mask.any():
                continue

            valid_p_diff = p_diff[valid_mask]
            valid_t_diff = t_diff[valid_mask]

            if len(valid_p_diff) > self.max_pairs_per_day:
                perm = torch.randperm(len(valid_p_diff), device=device)[:self.max_pairs_per_day]
                valid_p_diff = valid_p_diff[perm]
                valid_t_diff = valid_t_diff[perm]

            y_sign = torch.sign(valid_t_diff)
            loss = torch.nn.functional.softplus(-y_sign * valid_p_diff).mean()
            pairwise_losses.append(loss)

        if len(pairwise_losses) == 0:
            return mse_loss

        pairwise_loss = torch.stack(pairwise_losses).mean()

        if not hasattr(self, '_debug_printed'):
            print(f"  [Hybrid] MSE={mse_loss.item():.2f}, Pairwise={pairwise_loss.item():.4f}, "
                    f"Num dates with valid pairs={len(pairwise_losses)}")
            self._debug_printed = True

        return self.alpha * mse_loss + (1.0 - self.alpha) * pairwise_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, accumulation_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for i, (X, y, dates, _) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        
        with autocast():
            preds = model(X)
            if isinstance(criterion, HybridLoss):
                loss = criterion(preds, y, dates)
            else:
                loss = criterion(preds, y)
                
        scaler.scale(loss / accumulation_steps).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1 == len(dataloader)):
            # Unscale the gradients before clipping
            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
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
            with autocast():
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

    # 诊断信息：预测值的统计分布
    pred_std = all_preds.std().item()
    pred_mean = all_preds.mean().item()
    # 纯零预测 baseline
    mse_baseline = (all_targets ** 2).mean().item()
    mse_actual = nn.MSELoss()(all_preds, all_targets).item()

    print(f"  [Diag] Pred mean={pred_mean:.4f}, std={pred_std:.4f}")
    print(f"  [Diag] MSE baseline(zero pred)={mse_baseline:.2f}, MSE actual={mse_actual:.2f}")
    
    return avg_loss, ic

def train_model(data_path, train_period, val_period, save_path,
                seq_len=15, epochs=15, batch_size=24576, lr=1e-3,
                patience=5, target_col='label_return_1d',
                loss_type='mse', alpha=0.5, df=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Preparing Dataloaders for Train: {train_period}, Val: {val_period} with target: {target_col}...")
    train_loader, val_loader, _, num_features, _ = get_dataloaders(
        data_path, seq_len=seq_len,
        train_period=train_period, val_period=val_period, test_period=None,
        target_col=target_col, batch_size=batch_size
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
    scaler = GradScaler()
    
    best_val_ic = -float('inf')
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    print("Starting Training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
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
    data_path = '../../data/processed/800_stocks_features.parquet'
    train_model(
        data_path=data_path,
        train_period=('2016-01-01', '2024-12-31'),
        val_period=('2025-01-01', '2025-12-31'),
        save_path='checkpoints/best_ensemble.pth',
        target_col='label_return_5d',
        loss_type='hybrid',
        alpha=0.5,
        lr=1e-3
    )

if __name__ == '__main__':
    main()
