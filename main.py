"""
Main entry point for the stock prediction pipeline.
Usage: python main.py [--mode train|eval|backtest|all]
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch

from config import (
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    SEQ_LEN, TARGET_N, DEVICE, OUTPUT_DIR, N_HOLD, K_TRADE, SAMPLE_STEP,
)
from data_loader import load_all_data, load_basic_info
from config import MIN_LIST_DAYS, EXCLUDE_ST
from features import compute_all_features, get_feature_columns
from dataset import StockDataset, create_dataloaders, compute_cross_sectional_targets, collate_fn
from models import create_model, GRUAttention, MLPBaseline
from train import train_model
from backtest import BacktestEngine


def prepare_data():
    """Load data, compute features, split by time."""
    # Load raw data
    df_raw = load_all_data()

    # Load basic info
    basic_df = load_basic_info()

    # Split by time
    df_train = df_raw[
        (df_raw["trade_date"] >= TRAIN_START) & (df_raw["trade_date"] <= TRAIN_END)
    ].copy()
    df_val = df_raw[
        (df_raw["trade_date"] >= VAL_START) & (df_raw["trade_date"] <= VAL_END)
    ].copy()
    df_test = df_raw[
        (df_raw["trade_date"] >= TEST_START) & (df_raw["trade_date"] <= TEST_END)
    ].copy()

    print(f"[INFO] Train: {df_train['trade_date'].min()} ~ {df_train['trade_date'].max()}")
    print(f"[INFO] Val:   {df_val['trade_date'].min()} ~ {df_val['trade_date'].max()}")
    print(f"[INFO] Test:  {df_test['trade_date'].min()} ~ {df_test['trade_date'].max()}")

    # Apply filters BEFORE feature computation to save work
    if EXCLUDE_ST and "is_st" in df_train.columns:
        df_train = df_train[~df_train["is_st"]]
        df_val = df_val[~df_val["is_st"]]
        df_test = df_test[~df_test["is_st"]]

    def filter_min_days(d, min_d):
        return d.groupby("ts_code", group_keys=False).filter(lambda g: len(g) >= min_d)

    df_train = filter_min_days(df_train, MIN_LIST_DAYS)
    df_val = filter_min_days(df_val, MIN_LIST_DAYS)
    df_test = filter_min_days(df_test, MIN_LIST_DAYS)

    # Compute features ONCE on the full dataset (3x faster than per-split)
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    print(f"\n[INFO] Combined data: {len(df_all)} rows, computing features once...")
    df_all = compute_all_features(df_all, basic_df)

    # Compute cross-sectional targets on full data
    df_all = compute_cross_sectional_targets(df_all)

    # Split back
    train_end_dt = pd.Timestamp(TRAIN_END)
    val_end_dt = pd.Timestamp(VAL_END)
    df_train = df_all[df_all["trade_date"] <= train_end_dt].copy()
    df_val = df_all[(df_all["trade_date"] > train_end_dt) & (df_all["trade_date"] <= val_end_dt)].copy()
    df_test = df_all[df_all["trade_date"] > val_end_dt].copy()
    del df_all

    print(f"[INFO] After filtering+features: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # Save processed data
    df_train.to_parquet(os.path.join(OUTPUT_DIR, "train_processed.parquet"))
    df_val.to_parquet(os.path.join(OUTPUT_DIR, "val_processed.parquet"))
    df_test.to_parquet(os.path.join(OUTPUT_DIR, "test_processed.parquet"))

    feature_cols = get_feature_columns(df_train)
    with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"\n[INFO] Feature count: {len(feature_cols)}")
    print(f"[INFO] Feature columns: {feature_cols}")
    return df_train, df_val, df_test, feature_cols


def train_mode(df_train, df_val, feature_cols, model_name="gru_attention", sample_step=SAMPLE_STEP):
    """Train the model."""
    train_loader, val_loader = create_dataloaders(
        df_train, df_val, feature_cols, sample_step=sample_step
    )

    input_dim = len(feature_cols)
    model = create_model(model_name, input_dim)
    print(f"\n[INFO] Model: {model_name}")
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model, history = train_model(
        model, train_loader, val_loader, model_name=model_name,
    )
    return model, history


@torch.no_grad()
def predict_dataset(model, df, feature_cols, batch_size=2048):
    """Generate predictions for a full dataset."""
    model.eval()
    model = model.to(DEVICE)

    dataset = StockDataset(df, feature_cols)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    all_preds = []
    all_years = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        pred = model(X_batch).cpu().numpy().flatten()
        all_preds.extend(pred.tolist())
        all_years.extend(y_batch.cpu().numpy().flatten().tolist())

    # Map back to metadata
    indices = dataset.samples
    pred_df = pd.DataFrame({
        "trade_date": dataset.dates[indices],
        "ts_code": dataset.codes[indices],
        "score": all_preds,
        "target": all_years,
    })
    return pred_df


def evaluate_mode(model, df_val, df_test, feature_cols, model_name):
    """Evaluate model on val/test sets."""
    from scipy.stats import spearmanr

    results = {}
    for name, df in [("val", df_val), ("test", df_test)]:
        pred_df = predict_dataset(model, df, feature_cols)

        # Compute daily Rank IC
        ic_values = []
        for date in sorted(pred_df["trade_date"].unique()):
            day = pred_df[pred_df["trade_date"] == date]
            if len(day) >= 10:
                ic, _ = spearmanr(day["score"], day["target"])
                ic_values.append(ic)

        ic_mean = np.mean(ic_values) if ic_values else 0
        ic_std = np.std(ic_values) if ic_values else 0
        icir = ic_mean / ic_std if ic_std > 0 else 0

        print(f"\n[{name.upper()} Results]")
        print(f"  Rank IC Mean: {ic_mean:.4f}")
        print(f"  Rank IC Std:  {ic_std:.4f}")
        print(f"  ICIR:         {icir:.4f}")
        print(f"  IC > 0 ratio: {(np.array(ic_values) > 0).mean():.2%}")

        results[name] = {
            "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
            "predictions": pred_df,
        }

    return results


def backtest_mode(model, df_val, df_test, feature_cols, model_name):
    """Run backtest on val and test periods."""
    backtest_results = {}

    for name, df in [("val", df_val), ("test", df_test)]:
        print(f"\n[INFO] Running backtest on {name} set...")

        # Generate predictions
        pred_df = predict_dataset(model, df, feature_cols)

        # Prepare daily price data
        daily_prices = df[["trade_date", "ts_code", "open", "high", "low",
                            "close", "pct_chg"]].copy()

        # Run backtest
        engine = BacktestEngine(n_hold=N_HOLD, k_trade=K_TRADE)
        metrics = engine.run(pred_df, daily_prices)

        print(f"\n[{name.upper()} Backtest Results]")
        print(f"  Total Return:  {metrics['total_return']:.2%}")
        print(f"  Annual Return: {metrics['annual_return']:.2%}")
        print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate:      {metrics['win_rate']:.2%}")

        backtest_results[name] = metrics

        # Save backtest records
        records_df = pd.DataFrame(metrics["records"])
        records_df.to_csv(
            os.path.join(OUTPUT_DIR, f"backtest_{name}_{model_name}.csv"),
            index=False,
        )

    return backtest_results


def generate_submission(model, df_test, feature_cols):
    """Generate daily predictions for live trading."""
    pred_df = predict_dataset(model, df_test, feature_cols)

    # Get latest date's predictions
    latest_date = pred_df["trade_date"].max()
    latest_preds = pred_df[pred_df["trade_date"] == latest_date].sort_values(
        "score", ascending=False
    )
    print(f"\n[INFO] Latest predictions for {latest_date}:")
    print(latest_preds.head(N_HOLD * 2).to_string())

    pred_df.to_csv(os.path.join(OUTPUT_DIR, "daily_predictions.csv"), index=False)
    return pred_df


def main():
    parser = argparse.ArgumentParser(description="Stock Prediction Pipeline")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["data", "train", "eval", "backtest", "all"],
                        help="Which stage to run")
    parser.add_argument("--model", type=str, default="gru_attention",
                        choices=["mlp", "gru_attention", "transformer"],
                        help="Model architecture")
    parser.add_argument("--skip_prep", action="store_true",
                        help="Skip data preparation (use cached)")
    args = parser.parse_args()

    # ── Data Preparation ──
    if args.mode in ["data", "all"] and not args.skip_prep:
        df_train, df_val, df_test, feature_cols = prepare_data()
    else:
        print("[INFO] Loading cached processed data...")
        df_train = pd.read_parquet(os.path.join(OUTPUT_DIR, "train_processed.parquet"))
        df_val = pd.read_parquet(os.path.join(OUTPUT_DIR, "val_processed.parquet"))
        df_test = pd.read_parquet(os.path.join(OUTPUT_DIR, "test_processed.parquet"))
        with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)
        print(f"[INFO] Loaded {len(feature_cols)} features")

    # ── Training ──
    if args.mode in ["train", "all"]:
        model, history = train_mode(df_train, df_val, feature_cols, args.model)

    # ── Evaluation ──
    if args.mode in ["eval", "all"]:
        # Load best model
        ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"{args.model}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] Checkpoint not found: {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = create_model(args.model, len(feature_cols))
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded checkpoint from {ckpt_path} (Val IC: {ckpt.get('val_ic', 'N/A')})")

        results = evaluate_mode(model, df_val, df_test, feature_cols, args.model)

    # ── Backtest ──
    if args.mode in ["backtest", "all"]:
        ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"{args.model}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] Checkpoint not found: {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = create_model(args.model, len(feature_cols))
        model.load_state_dict(ckpt["model_state_dict"])

        bt_results = backtest_mode(model, df_val, df_test, feature_cols, args.model)

        # Generate latest predictions for live trading
        pred_df = generate_submission(model, df_test, feature_cols)


if __name__ == "__main__":
    main()
