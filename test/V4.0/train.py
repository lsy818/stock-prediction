import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats
from tqdm import tqdm
from dataset import get_dataloaders
from model import EnsembleAttentionGRU

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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for X, y, _, _ in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        preds = model(X)
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
        for X, y, _, _ in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            
            total_loss += loss.item()
            all_preds.append(preds)
            all_targets.append(y)
            
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    ic = calc_ic(all_preds, all_targets)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, ic

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    epochs = 15
    lr = 0.001
    patience = 5
    
    data_path = '../../data/processed/800_stocks_features.parquet'
    
    print("Preparing Dataloaders...")
    # Adjust batch_size for 800 stocks * 5 years = approx 1 million rows
    train_loader, val_loader, test_loader, num_features = get_dataloaders(
        data_path, seq_len=15, batch_size=8192)
    
    print("Initializing Model...")
    # Initialize Ensemble Attention GRU
    model = EnsembleAttentionGRU(
        input_size=num_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        num_models=3 # Ensemble of 3 GRUs
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    best_val_ic = -float('inf')
    epochs_no_improve = 0
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Starting Training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_ic = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val IC: {val_ic:.4f}")
        
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'checkpoints/best_ensemble.pth')
            print("  [*] Best Model Saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == '__main__':
    main()
