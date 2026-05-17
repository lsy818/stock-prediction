"""
Backtesting framework for the trading strategy.
Simulates daily rebalancing with T+1 constraint and realistic constraints.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import N_HOLD, K_TRADE, INITIAL_CAPITAL


class BacktestEngine:
    """
    Simple backtest engine for daily rebalancing strategy.

    Strategy:
        - Hold n stocks with equal weight
        - Each day, sell k lowest-scored holdings and buy k highest-scored stocks
        - T+1 constraint: stocks bought today cannot be sold today
    """

    def __init__(
        self,
        n_hold=N_HOLD,
        k_trade=K_TRADE,
        initial_capital=INITIAL_CAPITAL,
        commission=0.001,   # 0.1% commission
        slippage=0.001,     # 0.1% slippage
        consider_limit=True,  # consider price limit (涨跌停)
    ):
        self.n_hold = n_hold
        self.k_trade = k_trade
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.consider_limit = consider_limit

        # State
        self.capital = initial_capital
        self.holdings = {}     # ts_code -> shares
        self.cash = initial_capital
        self.daily_records = []  # list of daily snapshots

    def run(
        self,
        df_predictions: pd.DataFrame,
        daily_data: pd.DataFrame,
        start_date=None,
        end_date=None,
    ):
        """
        Run backtest over the date range.

        Args:
            df_predictions: DataFrame with columns [trade_date, ts_code, score]
                score = model prediction (higher = better expected return)
            daily_data: DataFrame with columns [trade_date, ts_code, close, open, pct_chg]
                Need at least close price for execution
            start_date, end_date: optional date range filter
        """
        # Prepare data
        pred = df_predictions.copy()
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        daily = daily_data.copy()
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])

        if start_date:
            pred = pred[pred["trade_date"] >= pd.Timestamp(start_date)]
            daily = daily[daily["trade_date"] >= pd.Timestamp(start_date)]
        if end_date:
            pred = pred[pred["trade_date"] <= pd.Timestamp(end_date)]
            daily = daily[daily["trade_date"] <= pd.Timestamp(end_date)]

        dates = sorted(pred["trade_date"].unique())

        # Price lookup: {date: {code: close, open, pct_chg}}
        price_dict = {}
        for d in dates:
            day_data = daily[daily["trade_date"] == d]
            price_dict[d] = {}
            for _, row in day_data.iterrows():
                price_dict[d][row["ts_code"]] = {
                    "close": row["close"],
                    "open": row.get("open", row["close"]),
                    "pct_chg": row.get("pct_chg", 0),
                }

        self.cash = self.initial_capital
        self.holdings = {}
        self.daily_records = []
        pending_buys = []  # stocks bought today (can't sell today)

        for i, date in enumerate(tqdm(dates, desc="Backtesting")):
            # Get today's predictions and prices
            today_pred = pred[pred["trade_date"] == date].copy()
            if len(today_pred) == 0:
                continue

            today_prices = price_dict.get(date, {})
            if not today_prices:
                continue

            # Build score map
            score_map = dict(zip(today_pred["ts_code"], today_pred["score"]))
            # Remove stocks without price data
            score_map = {k: v for k, v in score_map.items() if k in today_prices}

            if not score_map:
                continue

            # Step 1: Mark today's value (before trading)
            portfolio_value = self._compute_portfolio_value(today_prices)
            self.daily_records.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "cash": self.cash,
                "holdings": dict(self.holdings),
            })

            # Step 2: Clear yesterday's pending buys (they are now available to sell)
            pending_buys = []

            # Step 3: Sell k lowest-scored holdings
            held_codes = list(self.holdings.keys())
            # Score current holdings
            held_scores = {c: score_map.get(c, -np.inf) for c in held_codes}
            # Remove pending buys from sellable list
            sellable = {c: s for c, s in held_scores.items() if c not in pending_buys}

            # Sort by score ascending (sell worst first)
            to_sell = sorted(sellable.items(), key=lambda x: x[1])[:self.k_trade]

            for code, _ in to_sell:
                if code in self.holdings:
                    price_info = today_prices[code]
                    sell_price = price_info["close"]

                    # Check if can sell (not limit-down)
                    if self.consider_limit and price_info.get("pct_chg", 0) <= -9.9:
                        continue  # cannot sell at limit down

                    shares = self.holdings.pop(code)
                    proceeds = shares * sell_price * (1 - self.commission - self.slippage)
                    self.cash += proceeds

            # Step 4: Buy k highest-scored stocks (not already held)
            non_held_scores = {c: s for c, s in score_map.items() if c not in self.holdings}
            to_buy = sorted(non_held_scores.items(), key=lambda x: x[1], reverse=True)[:self.k_trade]

            if len(to_buy) > 0:
                # Equal weight allocation per new position
                target_positions = self.n_hold
                position_cash = self.cash / max(len(to_buy), 1)
                # Cap at available cash
                position_cash = min(position_cash, self.cash / len(to_buy))

                for code, _ in to_buy:
                    price_info = today_prices[code]
                    buy_price = price_info["close"]

                    # Check if can buy (not limit-up)
                    if self.consider_limit and price_info.get("pct_chg", 0) >= 9.9:
                        continue  # cannot buy at limit up

                    cost_per_share = buy_price * (1 + self.commission + self.slippage)
                    if cost_per_share <= 0 or position_cash <= 0:
                        continue

                    shares = int(position_cash / cost_per_share)
                    if shares > 0:
                        cost = shares * buy_price * (1 + self.commission + self.slippage)
                        if cost <= self.cash:
                            self.cash -= cost
                            self.holdings[code] = self.holdings.get(code, 0) + shares
                            pending_buys.append(code)

        # Record final value
        final_date = dates[-1] if len(dates) > 0 else None
        if final_date and final_date in price_dict:
            final_value = self._compute_portfolio_value(price_dict[final_date])
            self.daily_records.append({
                "date": final_date,
                "portfolio_value": final_value,
                "cash": self.cash,
                "holdings": dict(self.holdings),
            })

        return self._compute_metrics()

    def _compute_portfolio_value(self, prices):
        """Compute total portfolio value = cash + stock holdings."""
        stock_value = 0.0
        for code, shares in self.holdings.items():
            if code in prices:
                stock_value += shares * prices[code]["close"]
        return self.cash + stock_value

    def _compute_metrics(self):
        """Compute backtest performance metrics."""
        if len(self.daily_records) < 2:
            return {
                "total_return": 0, "annual_return": 0, "sharpe_ratio": 0,
                "max_drawdown": 0, "win_rate": 0, "records": self.daily_records,
            }

        values = [r["portfolio_value"] for r in self.daily_records]
        dates = [r["date"] for r in self.daily_records]

        values = np.array(values)
        initial = values[0]

        # Total return
        total_return = (values[-1] - initial) / initial

        # Daily returns
        daily_returns = np.diff(values) / values[:-1]

        # Annualized return (assuming ~250 trading days)
        n_days = len(daily_returns)
        if n_days > 0:
            total_growth = values[-1] / initial
            years = n_days / 250
            if total_growth > 0 and years > 0:
                annual_return = total_growth ** (1 / years) - 1
            else:
                annual_return = 0.0
        else:
            annual_return = 0.0

        # Sharpe ratio (risk-free rate = 2%)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess = daily_returns.mean() - 0.02 / 250
            sharpe = np.sqrt(250) * excess / daily_returns.std()
        else:
            sharpe = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / (peak + 1e-10)
        max_dd = drawdown.min()

        # Win rate
        win_rate = (daily_returns > 0).mean()

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "n_days": n_days,
            "records": self.daily_records,
            "dates": dates,
        }
        return metrics


def compute_daily_ic(df_predictions: pd.DataFrame, daily_data: pd.DataFrame):
    """
    Compute daily Rank IC between predictions and actual returns.
    """
    daily = daily_data.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    pred = df_predictions.copy()
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])

    # Compute actual forward return
    daily["fwd_return"] = daily.groupby("ts_code")["close"].transform(
        lambda x: x.shift(-1) / x - 1.0
    )

    ic_list = []
    dates = sorted(set(pred["trade_date"].unique()) & set(daily["trade_date"].unique()))

    for d in dates:
        p = pred[pred["trade_date"] == d][["ts_code", "score"]]
        a = daily[daily["trade_date"] == d][["ts_code", "fwd_return"]].dropna()
        merged = p.merge(a, on="ts_code")
        if len(merged) < 10:
            continue

        from scipy.stats import spearmanr
        ic, _ = spearmanr(merged["score"], merged["fwd_return"])
        ic_list.append({"date": d, "ic": ic})

    if not ic_list:
        return {"ic_mean": 0, "ic_std": 0, "icir": 0, "ic_positive_ratio": 0}

    ic_df = pd.DataFrame(ic_list)
    ic_mean = ic_df["ic"].mean()
    ic_std = ic_df["ic"].std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_positive_ratio = (ic_df["ic"] > 0).mean()

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_positive_ratio": ic_positive_ratio,
        "ic_series": ic_df,
    }
