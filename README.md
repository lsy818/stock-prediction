# 深度学习股票趋势预测与模拟交易

基于深度学习的 A 股股票趋势预测系统，使用 EnsembleAttentionGRU 模型 + HybridLoss 进行 T+5 收益率预测，并基于预测结果构建 Top-20 选股策略参与模拟交易比赛。

## 环境配置

```bash
conda create -n dl python=3.9
conda activate dl
pip install -r requirements.txt
```

## 项目结构

```
├── src/
│   ├── data_preprocessing.py  # 数据预处理：股票池构建、特征工程、标签构造
│   ├── dataset.py            # 滑动窗口数据集 + DataLoader（含防泄露标准化）
│   ├── model.py              # EnsembleAttentionGRU 模型（3×AttentionGRU 集成）
│   ├── train.py              # 训练循环 + HybridLoss + Early Stopping
│   ├── backtest.py           # 历史回测引擎（含 T+1/涨跌停/手续费等约束）
│   ├── predict_daily.py      # 每日预测（加载模型 + 最新数据）
│   ├── predict_latest.py     # 备选预测脚本（逐文件读取，更慢但更完整）
│   └── run_final.py          # 比赛最终训练（全量数据 2 epoch）
├── output/                   # 输出目录（处理后数据、模型权重、预测结果）
├── data/A股数据/              # 原始数据（不提交）
├── report/
|   ├── figures/              # 报告图片
│   ├── report.pdf            # 完整实验报告
├── requirements.txt
└── README.md
```

## 使用方式

```bash
# 1. 数据预处理（生成 800_stocks_features.parquet）
python src/data_preprocessing.py

# 2. 训练模型（实验阶段：15 epoch + early stopping）
python src/train.py

# 3. 历史回测（验证集 + 测试集）
python src/backtest.py

# 4. 生成每日预测（用于模拟交易）
python src/predict_daily.py

# 5. 比赛训练（全量数据 2 epoch）
python src/run_final.py
```

## 模型架构

**EnsembleAttentionGRU**：3 个独立 AttentionGRU 子模型集成，总参数 152,259。

```
Input: [batch, 15, 57]
       │
       ├── AttentionGRU-1 (GRU→Attention→MLP) ──→ pred_1
       ├── AttentionGRU-2 (GRU→Attention→MLP) ──→ pred_2
       ├── AttentionGRU-3 (GRU→Attention→MLP) ──→ pred_3
       │
       └── Mean(pred_1, pred_2, pred_3) ──→ Output
```

**HybridLoss**：MSE + Pairwise Ranking Loss（α=0.5），同时优化预测精度和截面排序质量。

## 回测结果

| 指标 | 验证集 (2025) | 测试集 (2026.1-5) |
|------|-------------|------------------|
| 总收益率 | +55.02% | +1.72% |
| 夏普比率 | 3.12 | 1.08 |
| 最大回撤 | -9.07% | -2.02% |

## 比赛结果

2026-06-01 ~ 2026-06-12，初始资金 ¥1,000,000。比赛策略按 T+5 预测视野执行：每 5 个交易日左右重新生成预测并调仓，目标持仓为 Top-20 等权且尽量满仓；实际比赛中 6 月 1 日建仓，6 月 5 日进行一次主要调仓。
- 最终收益率：约 -1.69%
- 详情见 `report/report.pdf`
