"""
Backtesting framework for daily rebalancing strategy.
Strategy: equal-weight hold n stocks, daily rotate k worst -> k best.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import N_HOLD, K_TRADE, INITIAL_CAPITAL


class BacktestEngine:
    """
    Daily rebalancing backtest engine.

    Strategy:
        Day 0: Equal-weight buy top-n highest-scored stocks
        Each day: Sell k lowest-scored holdings, buy k highest-scored non-held,
                  then rebalance to equal weight across all n holdings
        T+1 constraint enforced: stocks bought today cannot be sold tomorrow
    """

    def __init__(
        self,
        n_hold=N_HOLD,
        k_trade=K_TRADE,
        initial_capital=INITIAL_CAPITAL,
        commission=0.001,
        slippage=0.001,
        consider_limit=True,
    ):
        self.n_hold = n_hold
        self.k_trade = k_trade
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.consider_limit = consider_limit

    def run(self, df_predictions, daily_data, start_date=None, end_date=None):
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

        # Price lookup
        price_dict = {}
        for d in dates:
            day_data = daily[daily["trade_date"] == d]
            price_dict[d] = {}
            for _, row in day_data.iterrows():
                price_dict[d][row["ts_code"]] = {
                    "close": row["close"],
                    "pct_chg": row.get("pct_chg", 0),
                }

        self.cash = self.initial_capital
        self.holdings = {}       # code -> shares
        self.pending = set()     # codes bought today, can't sell tomorrow
        self.daily_records = []

        for i, date in enumerate(tqdm(dates, desc="Backtesting")):
            today_pred = pred[pred["trade_date"] == date]
            if len(today_pred) == 0:
                continue
            today_prices = price_dict.get(date, {})
            if not today_prices:
                continue

            # Score map for all stocks today
            score_map = dict(zip(today_pred["ts_code"], today_pred["score"]))
            score_map = {k: v for k, v in score_map.items() if k in today_prices}
            if not score_map:
                continue

            # ── Record value BEFORE trading ──
            pv = self._portfolio_value(today_prices)
            self.daily_records.append({
                "date": date, "portfolio_value": pv,
                "cash": self.cash, "n_holdings": len(self.holdings),
            })

            # ── Release yesterday's pending (now available to sell) ──
            self.pending = set()

            # ── Identify stocks to sell (limit-down cannot sell) ──
            held_scores = {}
            for code in self.holdings:
                if code not in self.pending and code in score_map:
                    held_scores[code] = score_map[code]

            to_sell = sorted(held_scores.items(), key=lambda x: x[1])[:self.k_trade]

            for code, _ in to_sell:
                if code not in self.holdings:
                    continue
                pinfo = today_prices.get(code)
                if not pinfo:
                    continue
                if self.consider_limit and pinfo.get("pct_chg", 0) <= -9.9:
                    continue
                sell_price = pinfo["close"]
                shares = self.holdings.pop(code)
                proceeds = shares * sell_price * (1 - self.commission - self.slippage)
                self.cash += proceeds

            # ── First day: buy n_hold stocks; subsequent days: refill to n_hold ──
            target_count = self.n_hold
            slots_to_fill = target_count - len(self.holdings)
            n_to_buy = max(slots_to_fill, 0)

            if i == 0:
                # Day 0: buy top n_hold stocks
                n_to_buy = self.n_hold

            non_held = {c: s for c, s in score_map.items() if c not in self.holdings}
            to_buy = sorted(non_held.items(), key=lambda x: x[1], reverse=True)[:n_to_buy]

            if len(to_buy) > 0:
                # Equal weight: each position gets cash / len(holdings_after_buy)
                total_positions_after = len(self.holdings) + len(to_buy)
                if total_positions_after == 0:
                    continue
                cash_per_stock = self.cash / len(to_buy)

                for code, _ in to_buy:
                    pinfo = today_prices.get(code)
                    if not pinfo:
                        continue
                    buy_price = pinfo["close"]
                    if self.consider_limit and pinfo.get("pct_chg", 0) >= 9.9:
                        continue
                    cost_ps = buy_price * (1 + self.commission + self.slippage)
                    if cost_ps <= 0:
                        continue
                    shares = int(min(cash_per_stock, self.cash) / cost_ps)
                    if shares <= 0:
                        continue
                    cost = shares * cost_ps
                    if cost <= self.cash:
                        self.cash -= cost
                        self.holdings[code] = self.holdings.get(code, 0) + shares
                        self.pending.add(code)

            # ── Rebalance to equal weight ──
            if len(self.holdings) > 0:
                total_value = self._portfolio_value(today_prices)
                target_value = total_value / len(self.holdings)
                for code, shares in list(self.holdings.items()):
                    pinfo = today_prices.get(code)
                    if not pinfo:
                        continue
                    curr_value = shares * pinfo["close"]
                    diff = target_value - curr_value
                    # Rebalance via partial buy/sell
                    price = pinfo["close"]
                    if diff > 50:  # underweight: buy more
                        extra_shares = int(diff / (price * (1 + self.commission + self.slippage)))
                        extra_cost = extra_shares * price * (1 + self.commission + self.slippage)
                        if extra_shares > 0 and extra_cost <= self.cash:
                            self.cash -= extra_cost
                            self.holdings[code] += extra_shares
                    elif diff < -50:  # overweight: sell some
                        sell_shares = min(int(-diff / (price * (1 - self.commission - self.slippage))),
                                          shares // 2)  # don't sell more than half
                        if sell_shares > 0:
                            proceeds = sell_shares * price * (1 - self.commission - self.slippage)
                            if self.holdings[code] > sell_shares:
                                self.holdings[code] -= sell_shares
                                self.cash += proceeds

        # Final value after last day
        if len(dates) > 0:
            final_date = dates[-1]
            if final_date in price_dict:
                pv = self._portfolio_value(price_dict[final_date])
                self.daily_records.append({
                    "date": final_date, "portfolio_value": pv,
                    "cash": self.cash, "n_holdings": len(self.holdings),
                })

        return self._compute_metrics()

    def _portfolio_value(self, prices):
        stock_value = 0.0
        for code, shares in self.holdings.items():
            if code in prices:
                stock_value += shares * prices[code]["close"]
        return self.cash + stock_value

    def _compute_metrics(self):
        if len(self.daily_records) < 2:
            return {"total_return": 0, "annual_return": 0, "sharpe_ratio": 0,
                    "max_drawdown": 0, "win_rate": 0, "records": self.daily_records}

        values = np.array([r["portfolio_value"] for r in self.daily_records])
        initial = values[0]
        total_return = (values[-1] - initial) / initial if initial > 0 else 0.0

        daily_returns = np.diff(values) / (values[:-1] + 1e-10)
        n_days = len(daily_returns)

        if n_days > 0 and initial > 0:
            years = n_days / 250
            total_growth = values[-1] / initial
            annual_return = total_growth ** (1 / years) - 1 if total_growth > 0 and years > 0 else 0.0
        else:
            annual_return = 0.0

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess = daily_returns.mean() - 0.02 / 250
            sharpe = np.sqrt(250) * excess / daily_returns.std()
        else:
            sharpe = 0.0

        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / (peak + 1e-10)
        max_dd = drawdown.min()
        win_rate = (daily_returns > 0).mean()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "n_days": n_days,
            "records": self.daily_records,
        }


def compute_daily_ic(df_predictions, daily_data):
    """Compute daily Rank IC between predictions and actual forward returns."""
    from scipy.stats import spearmanr
    pred = df_predictions.copy()
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    daily = daily_data.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily["fwd_return"] = daily.groupby("ts_code")["close"].transform(
        lambda x: x.shift(-1) / x - 1.0
    )

    dates = sorted(set(pred["trade_date"].unique()) & set(daily["trade_date"].unique()))
    ic_list = []
    for d in dates:
        p = pred[pred["trade_date"] == d][["ts_code", "score"]]
        a = daily[daily["trade_date"] == d][["ts_code", "fwd_return"]].dropna()
        merged = p.merge(a, on="ts_code")
        if len(merged) >= 10:
            ic, _ = spearmanr(merged["score"], merged["fwd_return"])
            ic_list.append({"date": d, "ic": ic})

    if not ic_list:
        return {"ic_mean": 0, "ic_std": 0, "icir": 0, "ic_positive_ratio": 0}
    ic_df = pd.DataFrame(ic_list)
    return {
        "ic_mean": ic_df["ic"].mean(),
        "ic_std": ic_df["ic"].std(),
        "icir": ic_df["ic"].mean() / (ic_df["ic"].std() + 1e-10),
        "ic_positive_ratio": (ic_df["ic"] > 0).mean(),
        "ic_series": ic_df,
    }
