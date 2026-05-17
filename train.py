"""
Training loop with early stopping, LR scheduling, and mixed precision.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast  # PyTorch 2.4+
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
from config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS,
    EARLY_STOP_PATIENCE, GRAD_CLIP_NORM, WARMUP_EPOCHS,
    USE_AMP, OUTPUT_DIR,
)


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self._get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr

    def _get_lr(self):
        epoch = self.current_epoch
        if epoch <= self.warmup_epochs:
            # Linear warmup
            return self.base_lrs[0] * epoch / max(1, self.warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


def compute_rank_ic(y_pred, y_true):
    """
    Compute Spearman rank correlation (Rank IC).
    For a batch that spans a single trading day.
    """
    y_pred = y_pred.detach().cpu().numpy().flatten()
    y_true = y_true.detach().cpu().numpy().flatten()

    if len(y_pred) < 2:
        return 0.0

    # Use scipy for reliability, but numpy-only for speed
    pred_rank = np.argsort(np.argsort(y_pred)).astype(np.float64)
    true_rank = np.argsort(np.argsort(y_true)).astype(np.float64)

    n = len(pred_rank)
    pred_rank_mean = pred_rank.mean()
    true_rank_mean = true_rank.mean()

    cov = ((pred_rank - pred_rank_mean) * (true_rank - true_rank_mean)).sum()
    std_pred = np.sqrt(((pred_rank - pred_rank_mean) ** 2).sum())
    std_true = np.sqrt(((true_rank - true_rank_mean) ** 2).sum())

    if std_pred < 1e-10 or std_true < 1e-10:
        return 0.0

    return cov / (std_pred * std_true)


def train_epoch(model, dataloader, optimizer, criterion, scaler, use_amp):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ic = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for X, y in pbar:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda"):
                pred = model(X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        ic = compute_rank_ic(pred, y)
        total_loss += loss.item()
        total_ic += ic
        n_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "ic": f"{ic:.4f}"})

    return total_loss / n_batches, total_ic / n_batches


@torch.no_grad()
def validate(model, dataloader, criterion):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    total_ic = 0.0
    n_batches = 0

    for X, y in tqdm(dataloader, desc="Validating", leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = criterion(pred, y)
        ic = compute_rank_ic(pred, y)

        total_loss += loss.item()
        total_ic += ic
        n_batches += 1

    return total_loss / n_batches, total_ic / n_batches


def train_model(
    model, train_loader, val_loader,
    model_name="gru_attention",
    max_epochs=MAX_EPOCHS,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
):
    """
    Full training loop with early stopping.
    Returns trained model and training history.
    """
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, max_epochs)
    scaler = GradScaler() if USE_AMP else None

    best_val_ic = -float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_ic": [], "val_loss": [], "val_ic": []}

    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{max_epochs}  LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_ic = train_epoch(model, train_loader, optimizer, criterion, scaler, USE_AMP)
        val_loss, val_ic = validate(model, val_loader, criterion)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_ic"].append(train_ic)
        history["val_loss"].append(val_loss)
        history["val_ic"].append(val_ic)

        print(f"Train Loss: {train_loss:.6f}  Train IC: {train_ic:.4f}")
        print(f"Val   Loss: {val_loss:.6f}  Val   IC: {val_ic:.4f}")

        # Early stopping based on validation IC
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_ic": val_ic, "val_loss": val_loss},
                os.path.join(checkpoint_dir, f"{model_name}_best.pt"),
            )
            print(f"  -> New best model saved (IC={val_ic:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Load best model
    best_ckpt = torch.load(
        os.path.join(checkpoint_dir, f"{model_name}_best.pt"),
        map_location=DEVICE, weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    print(f"\n[INFO] Best epoch: {best_epoch}, Best Val IC: {best_val_ic:.4f}")

    # Save history
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history
