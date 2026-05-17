### Workflow

0. 数据库更新
1. `data_preprocessing.py` 处理数据库，提取合并特征与标签
  - output: `data/processed/csi300_features.parquet`，包含特征与标签
2. `src/dataset.py` 读取预处理后的数据，构建数据集
  - output: `data/processed/train_dataset.pt`, `data/processed/val_dataset.pt`, `data/processed/test_dataset.pt`，包含特征与标签
3. `src/train.py` 训练模型
  - output: `src/results/best_gru.pth`
4. `src/backtest.py` 回测模型
  - output: `src/results/equity_curve.png`

