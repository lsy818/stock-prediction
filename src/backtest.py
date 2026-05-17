import os
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import StockGRU

def generate_predictions(model_path, data_path, seq_len=30, batch_size=512):
    """Run model over the validation set to get predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for prediction: {device}")
    
    # We only need val_loader for predictions
    _, val_loader, num_features = get_dataloaders(data_path, seq_len=seq_len, batch_size=batch_size)
    
    model = StockGRU(input_size=num_features, hidden_size=64, num_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    all_dates = []
    all_codes = []
    all_preds = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for X, _, dates, codes in tqdm(val_loader):
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

def backtest(df_preds, df_raw, initial_cash=1000000, top_k=10):
    """
    MVP Vectorized-like Backtester with real constraints.
    - Stamp duty: 0.05% (sell only)
    - Commission: 0.025% (buy & sell)
    - Lot size: 100 shares
    """
    print("Starting backtest...")
    df_raw['trade_date'] = pd.to_datetime(df_raw['trade_date'])
    df_raw_indexed = df_raw.set_index(['trade_date', 'ts_code'])[['open', 'close']]
    
    dates = sorted(df_preds['trade_date'].unique())
    
    cash = initial_cash
    holdings = {} # ts_code -> shares
    
    history = []
    trade_logs = []
    
    for i in tqdm(range(len(dates) - 1), desc="Backtesting days"):
        date_t = dates[i]
        date_next = dates[i+1]
        
        preds_t = df_preds[df_preds['trade_date'] == date_t]
        
        try:
            data_next = df_raw_indexed.loc[date_next]
            open_map = data_next['open'].to_dict()
            close_map = data_next['close'].to_dict()
        except KeyError:
            continue # Next day data missing
            
        # 1. Target Top K from day t's prediction
        target_df = preds_t.nlargest(top_k, 'pred')
        target_codes = set(target_df['ts_code'].tolist())
        
        # 2. Sell Phase at t+1 Open
        codes_to_sell = [code for code in holdings.keys() if code not in target_codes]
        for code in codes_to_sell:
            if code in open_map and not pd.isna(open_map[code]):
                shares = holdings[code]
                price = open_map[code]
                
                proceeds = shares * price * (1 - 0.00025 - 0.0005)
                cash += proceeds
                del holdings[code]
                trade_logs.append(f"{date_next.strftime('%Y-%m-%d')} | SELL | {code} | {shares} shares @ {price:.2f} | Proceeds: {proceeds:.2f}")

        # 3. Buy Phase at t+1 Open
        codes_to_buy = [code for code in target_codes if code not in holdings]
        if len(codes_to_buy) > 0:
            cash_per_stock = cash / len(codes_to_buy)
            for code in codes_to_buy:
                if code in open_map and not pd.isna(open_map[code]):
                    price = open_map[code]
                    cost_per_share = price * (1 + 0.00025)
                    shares = math.floor((cash_per_stock / cost_per_share) / 100) * 100
                    if shares > 0:
                        cost = shares * cost_per_share
                        if cash >= cost:
                            cash -= cost
                            holdings[code] = shares
                            trade_logs.append(f"{date_next.strftime('%Y-%m-%d')} | BUY  | {code} | {shares} shares @ {price:.2f} | Cost: {cost:.2f}")
                            
        # 4. Calculate Equity at t+1 Close
        stock_value = 0
        for code, shares in holdings.items():
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
    
    # Annualized Return (assuming 252 trading days)
    days = len(history_df)
    annualized_return = (1 + total_return) ** (252 / days) - 1
    
    # Max Drawdown
    roll_max = history_df['equity'].cummax()
    drawdown = history_df['equity'] / roll_max - 1
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio
    daily_rf = 0.03 / 252 # 3% annual risk free rate
    sharpe = (history_df['daily_return'].mean() - daily_rf) / (history_df['daily_return'].std() + 1e-8) * math.sqrt(252)
    
    print("\n" + "="*30)
    print("BACKTEST RESULTS (MVP)")
    print("="*30)
    print(f"Initial Equity: {initial_cash:.2f}")
    print(f"Final Equity:   {history_df['equity'].iloc[-1]:.2f}")
    print(f"Total Return:   {total_return*100:.2f}%")
    print(f"Annual Return:  {annualized_return*100:.2f}%")
    print(f"Max Drawdown:   {max_drawdown*100:.2f}%")
    print(f"Sharpe Ratio:   {sharpe:.2f}")
    print("="*30)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(history_df['trade_date'], history_df['equity'], label='Strategy Equity')
    plt.title('MVP Strategy Backtest (CSI 300 Components)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/equity_curve.png')
    print("Equity curve saved to results/equity_curve.png")
    
    # Save Trade Logs
    with open('results/trade_history.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(trade_logs))
    print("Trade history saved to results/trade_history.log")

if __name__ == '__main__':
    model_path = 'checkpoints/best_gru.pth'
    data_path = '../data/processed/csi300_features.parquet'
    
    # Generate predictions
    df_preds = generate_predictions(model_path, data_path)
    
    # Load raw data for prices
    print("Loading original prices for backtest...")
    df_raw = pd.read_parquet(data_path)
    
    # Run backtest
    backtest(df_preds, df_raw, top_k=10)
