### Workflow

0. 数据库更新

1. `data_preprocessing.py` 处理数据库，提取合并特征与标签
  - output: `data/processed/csi300_features.parquet`，包含特征与标签

2. `src/train.py` 训练模型
  - output: `src/results/best_*.pth`

3. `src/backtest.py` 回测模型
  - output: 收益率图表 `src/results/equity_curve.png`，交易记录 `src/results/trade_history.log`，结果信息 `src/results/backtest_summary.txt`

---

### Quantitative Strategy V4 Workflow

A-Share Quantitative Backtesting Pipeline (CSI 300 + CSI 500)

---

#### 1. Data Preprocessing (`test/V4/data_preprocessing.py`)
Extracts and merges price, financial, and capital flow data for CSI 300 and CSI 500 component stocks. Filters out ST/ST* warnings and Beijing Stock Exchange listings.
* **Input**: Raw daily CSV files in `data/daily/`, weight CSV files in `data/index_weight/`, industry listings, and ST warnings history.
* **Output**: `data/processed/800_stocks_features.parquet`, containing selected features, log-transformed market cap, and return labels.

#### 2. Single Run Training (`test/V4/train.py`)
Trains the final model on the defined train period and validates on the validation period for early stopping.
* **Training Window**: 2016-01-01 to 2024-12-31 (using board-specific start dates to align regime shifts).
* **Validation Window**: 2025-01-01 to 2025-12-31.
* **Model**: `EnsembleAttentionGRU` (Ensemble of 3 Local Attention GRU sub-models).
* **Output**: Final checkpoint saved to `test/V4/checkpoints/best_ensemble.pth`.

#### 3. Dual Backtesting (`test/V4/backtest.py`)
Simulates trading using the final saved model checkpoint. It runs on both the Validation Set (2025) and Test Set (2026) to compare out-of-sample performance and parameter sensitivity.
* **Trading Parameters**: Top 30 stocks ranked by model predictions, Open Price execution (to align with retail/non-algorithmic real-world order execution), and system-wide de-risking mode (max position capped at 30%) if CSI 300 index falls below its 60-day moving average (MA60).
* **Output**:
  * **Validation Set (2025)**:
    * `test/V4/results/equity_curve_验证集.png`
    * `test/V4/results/trade_history_验证集.log`
    * `test/V4/results/backtest_summary_验证集.txt`
  * **Test Set (2026)**:
    * `test/V4/results/equity_curve_测试集.png`
    * `test/V4/results/trade_history_测试集.log`
    * `test/V4/results/backtest_summary_测试集.txt`

#### 4. Time Series Cross-Validation (`test/V4/ts_cross_val.py`)
Executes Walk-Forward Time Series Cross-Validation (TSCV) over a sliding window across 3 Folds to evaluate the strategy's stability and avoid overfitting to a single market regime.
* **Folds Definition**:
  * **Fold 1**: Train (2016-2022) → Validate & Backtest (2023)
  * **Fold 2**: Train (2017-2023) → Validate & Backtest (2024)
  * **Fold 3**: Train (2018-2024) → Validate & Backtest (2025)
* **Output**:
  * Checkpoints: `test/V4/checkpoints/best_ensemble_fold_{1,2,3}.pth`
  * Backtest files: Individual curve, logs, and summaries for each fold's validation year in `test/V4/results/`.
  * CV Performance Report: `test/V4/results/ts_cross_val_summary.txt` (including average metrics across all folds).
