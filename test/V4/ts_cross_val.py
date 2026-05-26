import os
import gc
import pandas as pd
import numpy as np
import torch
from train import train_model
from backtest import generate_predictions, backtest

def main():
    data_path = '../../data/processed/800_stocks_features.parquet'

    # 3-Fold 滑动窗口时间序列划分
    time_windows = [
        # (train_start, train_end, val_start, val_end)
        ('2016-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),  # Fold 1
        ('2017-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),  # Fold 2
        ('2018-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),  # Fold 3
    ]

    cv_metrics = []

    # ========== 一次性加载全量数据 ==========
    # df_full: 保留全部原始列，用于回测（open/close/vwap/pre_close/volatility20 + label列）
    # df_model: 经板块过滤后的数据，用于训练和预测，避免每个 Fold 重复读盘
    print("Loading full parquet data (once for all folds)...")
    df_full = pd.read_parquet(data_path)
    df_full['trade_date'] = pd.to_datetime(df_full['trade_date'])

    # 板块过滤（创业板 2020-08-24 前数据剔除，其他板块保留 2016 年起）
    mask_chinext = df_full['ts_code'].str.startswith('300')
    df_model = pd.concat([
        df_full[mask_chinext & (df_full['trade_date'] >= '2020-08-24')],
        df_full[~mask_chinext & (df_full['trade_date'] >= '2016-01-01')]
    ], ignore_index=True)
    df_model = df_model.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    print(f"Full data: {len(df_full)} rows | Model-ready (filtered): {len(df_model)} rows")

    for i, (train_start, train_end, val_start, val_end) in enumerate(time_windows):
        fold_label = f"Fold_{i+1}_val"
        checkpoint_path = f'checkpoints/best_ensemble_fold_{i+1}.pth'

        print("\n" + "="*50)
        print(f"=== Fold {i+1}/3: Train ({train_start} ~ {train_end}) | Val ({val_start} ~ {val_end}) ===")
        print("="*50)

        # 1. 训练模型（传入预加载的 df_model，跳过重复读盘）
        print(f"Training model for Fold {i+1}...")
        model = train_model(
            data_path=data_path,
            train_period=(train_start, train_end),
            val_period=(val_start, val_end),
            save_path=checkpoint_path,
            epochs=15,
            batch_size=49152,
            lr=1e-3,
            patience=5,
            target_col='label_return_5d',
            loss_type='hybrid',
            alpha=0.5,
            df=df_model
        )

        # 显式释放训练模型占用的 GPU 显存
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # 2. 预测验证集（复用同一个 df_model，不再重复读盘）
        print(f"Generating predictions for Fold {i+1} validation period...")
        df_preds = generate_predictions(
            model_path=checkpoint_path,
            data_path=data_path,
            seq_len=15,
            dataset_type='val',
            train_period=(train_start, train_end),
            val_period=(val_start, val_end),
            target_col='label_return_5d',
            df=df_model
        )

        # 3. 回测评估（使用完整的 df_full，包含价格列）
        print(f"Running backtest for Fold {i+1} validation period...")
        metrics = backtest(
            df_preds=df_preds,
            df_raw=df_full,
            top_k=30,
            label=fold_label,
            ic_label='label_return_5d',
        )

        metrics['fold'] = i + 1
        metrics['train_period'] = f"{train_start}~{train_end}"
        metrics['val_period'] = f"{val_start}~{val_end}"
        cv_metrics.append(metrics)

        # 释放预测结果内存，防止 Fold 间累积
        del df_preds
        torch.cuda.empty_cache()
        gc.collect()
        
    # 汇总输出
    print("\n" + "="*50)
    print("=== TIME SERIES CROSS VALIDATION SUMMARY ===")
    print("="*50)
    
    summary_lines = []
    summary_lines.append(f"{'Fold':<6} | {'Train Period':<23} | {'Val Period':<23} | {'Total Ret':<10} | {'Ann Ret':<9} | {'Max DD':<9} | {'Sharpe':<8} | {'Win Rate':<9} | {'Rank IC':<8} | {'ICIR':<6}")
    summary_lines.append("-" * 125)
    
    total_rets = []
    ann_rets = []
    max_dds = []
    sharpes = []
    win_rates = []
    ics = []
    icirs = []
    
    for m in cv_metrics:
        total_rets.append(m['total_return'])
        ann_rets.append(m['annualized_return'])
        max_dds.append(m['max_drawdown'])
        sharpes.append(m['sharpe'])
        win_rates.append(m['trade_win_rate'])
        ics.append(m['ic_mean'])
        icirs.append(m['ic_ir'])
        
        summary_lines.append(
            f"{m['fold']:<6} | {m['train_period']:<23} | {m['val_period']:<23} | "
            f"{m['total_return']*100:>8.2f}% | {m['annualized_return']*100:>7.2f}% | {m['max_drawdown']*100:>7.2f}% | "
            f"{m['sharpe']:>8.2f} | {m['trade_win_rate']*100:>7.2f}% | {m['ic_mean']:>8.4f} | {m['ic_ir']:>6.4f}"
        )
        
    summary_lines.append("-" * 125)
    summary_lines.append(
        f"{'Avg':<6} | {'-'*23:<23} | {'-'*23:<23} | "
        f"{np.mean(total_rets)*100:>8.2f}% | {np.mean(ann_rets)*100:>7.2f}% | {np.mean(max_dds)*100:>7.2f}% | "
        f"{np.mean(sharpes):>8.2f} | {np.mean(win_rates)*100:>7.2f}% | {np.mean(ics):>8.4f} | {np.mean(icirs):>6.4f}"
    )
    summary_lines.append("="*125)
    
    summary_report = "\n".join(summary_lines)
    print(summary_report)
    
    os.makedirs('results', exist_ok=True)
    with open('results/ts_cross_val_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)
        
    print("\nCross-validation summary saved to results/ts_cross_val_summary.txt")

if __name__ == '__main__':
    main()
