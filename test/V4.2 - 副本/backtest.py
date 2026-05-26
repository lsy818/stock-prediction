import os
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import EnsembleAttentionGRU

def is_limit_up(code, open_price, pre_close):
    limit_ratio = 1.20 if code.startswith('688') or code.startswith('300') else 1.10
    limit_price = round(pre_close * limit_ratio, 2)
    return open_price >= limit_price - 1e-4

def is_limit_down(code, open_price, pre_close):
    limit_ratio = 0.80 if code.startswith('688') or code.startswith('300') else 0.90
    limit_price = round(pre_close * limit_ratio, 2)
    return open_price <= limit_price + 1e-4

def generate_predictions(model_path, data_path, seq_len=15, batch_size=512, dataset_type='test',
                         train_period=('2016-01-01', '2024-12-31'),
                         val_period=('2025-01-01', '2025-12-31'),
                         test_period=('2026-01-01', '2026-12-31'),
                         target_col='label_return_5d'):
    """Run model over the validation or test set to get predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for prediction: {device}")
    
    # Select correct loader based on dataset_type
    if dataset_type == 'val':
        _, val_loader, _, num_features = get_dataloaders(
            data_path, seq_len=seq_len, batch_size=batch_size, 
            train_period=train_period, val_period=val_period, test_period=None,
            target_col=target_col
        )
        loader = val_loader
    else:
        _, _, test_loader, num_features = get_dataloaders(
            data_path, seq_len=seq_len, batch_size=batch_size, 
            train_period=train_period, val_period=None, test_period=test_period,
            target_col=target_col
        )
        loader = test_loader
    
    model = EnsembleAttentionGRU(
        input_size=num_features, 
        hidden_size=64, 
        num_layers=2, 
        dropout=0.2,
        num_models=3
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    all_dates = []
    all_codes = []
    all_preds = []
    
    print(f"Generating predictions for {dataset_type} set...")
    with torch.no_grad():
        for X, _, dates, codes in tqdm(loader):
            X = X.to(device)
            preds = model(X).cpu().numpy()
            
            all_dates.extend(dates)
            all_codes.extend(codes)
            all_preds.extend(preds)
            
    df_preds = pd.DataFrame({
        'trade_date': all_dates,
        'ts_code': all_codes,
        'pred': all_preds
    })
    
    # Dates are returned as strings, ensure datetime
    df_preds['trade_date'] = pd.to_datetime(df_preds['trade_date'])
    return df_preds

def backtest(df_preds, df_raw, initial_cash=1000000, top_k=30, sell_threshold=60, ic_label='label_return_5d', label="测试集"):
    """
    V4 Vectorized-like Backtester with VWAP execution and Macro MA60 Filter.
    """
    print("Starting backtest...")
    df_raw['trade_date'] = pd.to_datetime(df_raw['trade_date'])
    df_raw_indexed = df_raw.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close', 'vwap', 'volatility20']]
    
    # Load CSI 300 for Macro Filter
    try:
        index_df = pd.read_csv('../../data/market/000300.SH.csv')
        index_df['trade_date'] = pd.to_datetime(index_df['trade_date'].astype(str))
        index_df = index_df.sort_values('trade_date')
        index_df['ma60'] = index_df['close'].rolling(60).mean()
        index_map = index_df.set_index('trade_date')[['close', 'ma60']].to_dict('index')
        print("Successfully loaded CSI 300 index for macro filtering.")
    except Exception as e:
        print(f"Warning: Could not load index for macro filter: {e}")
        index_map = {}
        
    dates = sorted(df_preds['trade_date'].unique())
    
    cash = initial_cash
    holdings = {} # ts_code -> {'shares': int, 'cost_price': float}
    
    history = []
    trade_logs = []
    trade_results = [] # True for win, False for loss
    
    cash_buffer_ratio = 0.995 # Leave 0.5% cash for fees
    
    for i in tqdm(range(len(dates) - 1), desc="Backtesting days"):
        date_t = dates[i]
        date_next = dates[i+1]
        
        preds_t = df_preds[df_preds['trade_date'] == date_t]
        
        try:
            data_next = df_raw_indexed.loc[date_next]
            open_map = data_next['open'].to_dict()
            close_map = data_next['close'].to_dict()
            pre_close_map = data_next['pre_close'].to_dict()
            vwap_map = data_next['vwap'].to_dict()
        except KeyError:
            continue # Next day data missing
            
        # Macro Timing Filter Check
        max_pos_ratio = 1.0
        if date_t in index_map:
            idx_close = index_map[date_t]['close']
            idx_ma60 = index_map[date_t]['ma60']
            if not pd.isna(idx_ma60) and idx_close < idx_ma60:
                max_pos_ratio = 0.3 # Reduce total position to 30% if below MA60
                
        # Sort predictions of day t
        preds_sorted = preds_t.sort_values('pred', ascending=False)
        rank_list = list(preds_sorted['ts_code'])
        rank_map = {code: idx + 1 for idx, code in enumerate(rank_list)}
        pred_map = {row['ts_code']: row['pred'] for _, row in preds_sorted.iterrows()}
        
        # 1. Check current holdings to see which ones we keep
        keep_codes = []
        for code in list(holdings.keys()):
            rank_val = rank_map.get(code, 999)
            pred_val = pred_map.get(code, 0.0)
            # Keep if ranking is within sell_threshold and prediction is positive
            if rank_val <= sell_threshold and pred_val > 0:
                keep_codes.append(code)
                
        # 2. Fill empty slots with new buys from the top ranks
        empty_slots = top_k - len(keep_codes)
        buy_codes = []
        if empty_slots > 0:
            for code in rank_list:
                if len(buy_codes) >= empty_slots:
                    break
                # Only buy if it's not already kept, and has positive prediction
                if code not in keep_codes and pred_map.get(code, 0.0) > 0:
                    buy_codes.append(code)
                    
        valid_codes = keep_codes + buy_codes
            
        target_weights = {}
        if len(valid_codes) > 0:
            weight_per_stock = 1.0 / len(valid_codes)
            target_weights = {code: weight_per_stock for code in valid_codes}
            
        # 3. Portfolio Rebalancing at t+1 Open
        current_equity = cash
        for code, holding_data in holdings.items():
            shares = holding_data['shares']
            price = open_map.get(code, holding_data['cost_price'])
            current_equity += shares * price
            
        # Apply Macro Position Limit
        target_equity_to_allocate = current_equity * max_pos_ratio * cash_buffer_ratio
        
        # Calculate target shares
        target_shares_map = {}
        for code, weight in target_weights.items():
            target_value = target_equity_to_allocate * weight
            if code in open_map and not pd.isna(open_map[code]):
                price = open_map[code]
                cost_per_share = price * (1 + 0.00025)
                
                if code.startswith('688'):
                    shares = math.floor(target_value / cost_per_share)
                    if shares < 200: shares = 0
                else:
                    shares = math.floor((target_value / cost_per_share) / 100) * 100
                    
                if shares > 0:
                    target_shares_map[code] = shares
                    
        # Sell Phase
        codes_held = list(holdings.keys())
        for code in codes_held:
            current_shares = holdings[code]['shares']
            target_shares = target_shares_map.get(code, 0)
            
            if current_shares > target_shares:

                shares_to_sell = current_shares - target_shares
                
                # [修复] 检查单笔卖出单是否合规（除非清仓 target_shares == 0）
                if target_shares != 0:
                    if code.startswith('688') and shares_to_sell < 200:
                        target_shares_map[code] = current_shares # 达不到最低卖出门槛，放弃调仓
                        continue
                    elif code.startswith('300') and shares_to_sell < 100:
                        target_shares_map[code] = current_shares # 达不到最低卖出门槛，放弃调仓
                        continue
                    elif not code.startswith('688') and not code.startswith('300'):
                        if shares_to_sell % 100 != 0:
                            # 主板调仓向下取整到 100 的倍数
                            shares_to_sell = math.floor(shares_to_sell / 100) * 100
                            if shares_to_sell <= 0:
                                target_shares_map[code] = current_shares
                                continue
                            target_shares = current_shares - shares_to_sell
                            target_shares_map[code] = target_shares


                if code in open_map and not pd.isna(open_map[code]) and not pd.isna(pre_close_map.get(code)):
                    open_price = open_map[code]
                    pre_close = pre_close_map[code]
                    
                    if is_limit_down(code, open_price, pre_close):
                        trade_logs.append(f"{date_next.strftime('%Y-%m-%d')} | HOLD | {code} | Limit Down, Sell Rejected.")
                        target_shares_map[code] = current_shares
                        continue
                        
                    proceeds = shares_to_sell * open_price * (1 - 0.00025 - 0.0005)
                    cash += proceeds
                    
                    if target_shares == 0:
                        cost_price = holdings[code]['cost_price']
                        total_cost = shares_to_sell * cost_price * (1 + 0.00025)
                        trade_results.append(proceeds > total_cost)
                        del holdings[code]
                    else:
                        holdings[code]['shares'] = target_shares
                        
                    trade_logs.append(f"{date_next.strftime('%Y-%m-%d')} | SELL | {code} | {shares_to_sell} shares @ Open {open_price:.2f} | Proceeds: {proceeds:.2f}")

        # Buy Phase
        for code, target_shares in list(target_shares_map.items()):
            current_shares = holdings.get(code, {}).get('shares', 0)
            if target_shares > current_shares:
                
                shares_to_buy = target_shares - current_shares
                
                # [修复] 检查单笔加仓买入单是否合规
                if code.startswith('688') and shares_to_buy < 200:
                    continue # 放弃这笔微小加仓
                elif code.startswith('300') and shares_to_buy < 100:
                    continue # 放弃这笔微小加仓
                elif not code.startswith('688') and not code.startswith('300'):
                    shares_to_buy = math.floor(shares_to_buy / 100) * 100
                    if shares_to_buy <= 0:
                        continue
                    target_shares = current_shares + shares_to_buy
                
                if code in open_map and not pd.isna(open_map[code]) and not pd.isna(pre_close_map.get(code)):
                    open_price = open_map[code]
                    pre_close = pre_close_map[code]
                    
                    if is_limit_up(code, open_price, pre_close):
                        trade_logs.append(f"{date_next.strftime('%Y-%m-%d')} | SKIP | {code} | Limit Up, Buy Rejected.")
                        target_shares_map[code] = current_shares
                        continue
                        
                    cost_per_share = open_price * (1 + 0.00025)
                    cost = shares_to_buy * cost_per_share
                    
                    if cash < cost:
                        if code.startswith('688'):
                            affordable_shares = math.floor(cash / cost_per_share)
                            if affordable_shares < 200: 
                                affordable_shares = 0
                        else:
                            affordable_shares = math.floor((cash / cost_per_share) / 100) * 100
                        shares_to_buy = affordable_shares
                        cost = shares_to_buy * cost_per_share
                        target_shares = current_shares + shares_to_buy
                        
                    if shares_to_buy > 0:
                        cash -= cost
                        if code in holdings:
                            old_shares = holdings[code]['shares']
                            old_cost = holdings[code]['cost_price']
                            new_cost_price = (old_shares * old_cost + shares_to_buy * open_price) / target_shares
                            holdings[code]['shares'] = target_shares
                            holdings[code]['cost_price'] = new_cost_price
                        else:
                            holdings[code] = {'shares': target_shares, 'cost_price': open_price}
                        
                        trade_logs.append(f"{date_next.strftime('%Y-%m-%d')} | BUY  | {code} | {shares_to_buy} shares @ Open {open_price:.2f} | Cost: {cost:.2f}")
                            
        # 4. Calculate Equity at t+1 Close
        stock_value = 0
        for code, holding_data in holdings.items():
            shares = holding_data['shares']
            if code in close_map and not pd.isna(close_map[code]):
                stock_value += shares * close_map[code]
            elif code in open_map and not pd.isna(open_map[code]):
                stock_value += shares * open_map[code]
                
        equity = cash + stock_value
        history.append({'trade_date': date_next, 'equity': equity, 'cash': cash})
                        
    history_df = pd.DataFrame(history)
    
    # Calculate Metrics
    history_df['daily_return'] = history_df['equity'].pct_change()
    total_return = history_df['equity'].iloc[-1] / initial_cash - 1
    
    days = len(history_df)
    annualized_return = (1 + total_return) ** (252 / days) - 1
    
    roll_max = history_df['equity'].cummax()
    drawdown = history_df['equity'] / roll_max - 1
    max_drawdown = drawdown.min()
    
    daily_rf = 0.03 / 252 
    sharpe = (history_df['daily_return'].mean() - daily_rf) / (history_df['daily_return'].std() + 1e-8) * math.sqrt(252)
    
    trade_win_rate = sum(trade_results) / len(trade_results) if len(trade_results) > 0 else 0
    daily_win_rate = (history_df['daily_return'] > 0).mean()
    
    print("Calculating IC/ICIR...")
    ic_df = pd.merge(df_preds, df_raw[['trade_date', 'ts_code', ic_label]], on=['trade_date', 'ts_code'], how='inner')
    ic_series = ic_df.groupby('trade_date').apply(lambda x: x['pred'].corr(x[ic_label], method='spearman'))
    ic_mean = ic_series.mean()
    ic_ir = ic_mean / ic_series.std() if ic_series.std() != 0 else 0
    
    report_lines = [
        "="*30,
        f"BACKTEST RESULTS (V4 - Focus on Quality - {label} - Sell Threshold: {sell_threshold})",
        "="*30,
        f"Initial Equity:    {initial_cash:.2f}",
        f"Final Equity:      {history_df['equity'].iloc[-1]:.2f}",
        f"Total Return:      {total_return*100:.2f}%",
        f"Annual Return:     {annualized_return*100:.2f}%",
        f"Max Drawdown:      {max_drawdown*100:.2f}%",
        f"Sharpe Ratio:      {sharpe:.2f}",
        f"Trade Win Rate:    {trade_win_rate*100:.2f}%",
        f"Daily Win Rate:    {daily_win_rate*100:.2f}%",
        f"Rank IC (Mean):    {ic_mean:.4f}",
        f"ICIR:              {ic_ir:.4f}",
        "="*30
    ]
    
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history_df['trade_date'], history_df['equity'], label='Strategy Equity')
    plt.title(f'Strategy Backtest on {label} Set (V4)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    
    file_suffix = f"_{label}" if label else ""
    plt.savefig(f'results/equity_curve_{ic_label}{file_suffix}.png')
    
    with open(f'results/trade_history_{ic_label}{file_suffix}.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(trade_logs))
        
    with open(f'results/backtest_summary_{ic_label}{file_suffix}.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    print(f"Equity curve saved to results/equity_curve_{ic_label}{file_suffix}.png")
    print(f"Trade history saved to results/trade_history_{ic_label}{file_suffix}.log")
    print(f"Summary report saved to results/backtest_summary_{ic_label}{file_suffix}.txt")
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trade_win_rate': trade_win_rate,
        'daily_win_rate': daily_win_rate,
        'ic_mean': ic_mean,
        'ic_ir': ic_ir
    }

if __name__ == '__main__':
    model_path = 'checkpoints/best_ensemble.pth'
    data_path = '../../data/processed/800_stocks_features.parquet'
    
    print("Loading original prices for backtest...")
    df_raw = pd.read_parquet(data_path)
    
    # Run Validation Set Backtest
    print("\n" + "="*40)
    print("=== Phase 1: Validation Set Backtest (2025) ===")
    print("="*40)
    df_preds_val = generate_predictions(model_path, data_path, seq_len=15, dataset_type='val', target_col='label_return_5d')
    backtest(df_preds_val, df_raw, top_k=30, ic_label='label_return_5d', label="验证集")
    
    # Run Test Set Backtest
    print("\n" + "="*40)
    print("=== Phase 2: Test Set Backtest (2026) ===")
    print("="*40)
    df_preds_test = generate_predictions(model_path, data_path, seq_len=15, dataset_type='test', target_col='label_return_5d')
    backtest(df_preds_test, df_raw, top_k=30, ic_label='label_return_5d', label="测试集")
