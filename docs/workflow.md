### Workflow

0. 数据库更新
1. `data_preprocessing.py` 处理数据库，提取合并特征与标签
  - output: `data/processed/csi300_features.parquet`，包含特征与标签
2. `src/train.py` 训练模型
  - output: `src/results/best_gru.pth`
3. `src/backtest.py` 回测模型
  - output: 收益率图表 `src/results/equity_curve.png`，交易记录 `src/results/trade_history.log`

