# 深度学习股票趋势预测

基于深度学习的 A 股股票趋势预测与模拟交易系统。

## 环境配置

```bash
conda activate dl
pip install -r requirements.txt
```

## 项目结构

```
├── config.py          # 全局配置
├── data_loader.py     # 数据加载与预处理
├── features.py        # 特征工程
├── dataset.py         # 滑动窗口数据集
├── models.py          # 模型架构（GRU+Attention / MLP / Transformer）
├── train.py           # 训练循环
├── backtest.py        # 回测引擎
├── main.py            # 主入口
├── analysis_report.md # 数据分析与方法设计文档
└── output/            # 输出目录（缓存数据、模型、结果）
```

## 使用方式

```bash
# 完整流程（数据准备 + 训练 + 评估 + 回测）
python main.py --mode all

# 仅数据准备
python main.py --mode data

# 仅训练
python main.py --mode train --model gru_attention

# 仅评估 + 回测
python main.py --mode backtest --model gru_attention
```

## 模型选择

- `mlp`: 简单 MLP 基线
- `gru_attention`: GRU + Multi-Head Attention（推荐）
- `transformer`: Transformer Encoder

## 预测输出

训练完成后，`output/daily_predictions.csv` 包含每日每只股票的预测分数，可用于模拟交易。
