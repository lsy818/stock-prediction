基于当前实验结果，以下是按优先级排序的改进方向。

---
方向一：将 IC 评估目标与训练目标对齐（投入极小）**Finished**

当前 `backtest.py` 第 333 行始终用 `label_return_1d` 计算 Rank IC。模型优化 T+5 收益，却用 T+1 收益评估排序质量——这个错配对 IC 的压制在 20-40%。

当前：模型预测 T+5 → 但与 T+1 收益算 IC → IC=0.022
对齐：模型预测 T+5 → 与 T+5 收益算 IC → 预计 IC=0.04~0.06

改动只需要让 IC 计算的 label 列可配置。这不仅让评估更诚实，还能帮你判断"IC 弱"到底是模型不行还是评估错了。

---
方向二：引入截面特征（投入小、效果确定）

当前模型只看个股自身的历史序列，完全不知道同日其他股票的表现。股市的核心 alpha 来源之一是相对强弱——一只股票涨 2%，如果板块涨 5%，它其实在走弱。

建议新增特征：

| 特征 | 计算方式 | 含义 | 备注 | 
| ---- | ------- | ---- | ---- |
| cross_rank_pct_chg │ 当日该股涨跌幅在全市场的分位数 (0~1) │ 截面相对强弱 | |
│ cross_rank_vol │ 当日成交量分位数 │ 是否为市场焦点 | 大盘股成交量天然高，考虑用换手率代替 | 
| cross_rank_turnover_rate | | 换手率分位数 | 换手率是归一化的，跨股票可比 |
│ cross_rank_mom5 │ 5 日动量分位数 │ 中期相对强弱 | |
│ industry_rel_return │ 个股收益 − 同行业平均收益 │ 行业内 alpha | | 
│ industry_mom │ 同行业近 5 日平均收益 │ 板块动量 | |
| volume_ratio | 当日量 / 5 日均量 | 量比 | 已经是截面可比的相对指标 |


实现方式是事后填充——data_preprocessing.py 中按日期 groupby 计算截面 rank 即可，不引入 look-ahead。这 5 个特征加到现有的 ~50 个特征中，几乎零额外成本，但提供了模型当前完全缺失的信息维度。

---
方向三：改进宏观择时（投入中）

当前宏观过滤只有一条规则：CSI 300 低于 MA60 → 仓位降到 30%。这太粗糙了。

建议的多维度择时：

```python
# 1. 波动率维度：市场 VIX 类指标
index_vol_20 = index_df['close'].pct_change().rolling(20).std()
if index_vol_20 > index_vol_20.rolling(60).mean() * 1.5:
    position_ratio *= 0.7  # 高波降低仓位

# 2. 成交量维度：缩量熊市和放量熊市不同对待
volume_ratio = index_df['vol'].rolling(20).mean() / index_df['vol'].rolling(60).mean()
if index_close < index_ma60 and volume_ratio < 0.8:
    position_ratio = 0  # 缩量阴跌→直接空仓

# 3. 连续下跌保护
if consecutive_down_days >= 5:
    position_ratio = 0.2  # 连跌 5 天→极度防御
```

2023 年 Fold 1 的 Max DD 只有 -7.26%，说明当前宏观过滤在震荡市已经有效。但 2024-2025 的 Max DD 仍在 -11~-15% 区间，更精细的择时可以进一步压低回撤。

---
方向四：处理标签的自相关性（投入小）

label_return_5d 有一个隐蔽问题：相邻两天（t 和 t+1）的 T+5 收益有 4 天重叠。

t 的 T+5:  day(t+1) + day(t+2) + day(t+3) + day(t+4) + day(t+5)
t+1的T+5:  day(t+2) + day(t+3) + day(t+4) + day(t+5) + day(t+6)
重叠 80%

这意味着连续两天的 label 高度自相关。模型可能学到"昨天预测高的股票今天继续预测高"，造成持仓几乎不变——降低了换手率，但也可能让模型惰性地持有已失效的股票。

解决方案：改用非重叠标签

```python
# 改为每隔 5 天计算一次标签（周五收盘后预测下周一~下周五累计收益）
# 回测仍每日调仓，但 label 不再有重叠
df['label_return_5d_non_overlap'] = df.groupby('ts_code')['pct_chg'].shift(-1).rolling(5).sum().shift(-4)
# 然后在 dataset 中只保留每周五的样本用于训练
```

或者更简单的替代方案：把回测频率从日频降为周频。模型每周末选出 top 30，持有整整一周。这天然对齐 T+5 的预测目标，且大幅降低交易成本。

---
方向五：分层建模（投入中高）

当前 800 只股票用一个模型。但沪深 300 大盘股和中证 500 中小盘股的行为模式差异巨大。

方案：

```python
# 训练两个独立模型
model_large = EnsembleAttentionGRU(...)  # CSI 300 专用
model_small = EnsembleAttentionGRU(...)  # CSI 500 专用

# 回测时各自选出 top 15，合并为 30 只
top_large = preds_large.nlargest(15, 'pred')
top_small = preds_small.nlargest(15, 'pred')
final_pool = pd.concat([top_large, top_small])
```

好处：
- 每个模型的特征分布更均匀（大盘股/小盘股的波动率、流动性、估值水平差异不会被 mix 在一起）
- 持仓分散天然分散风险

成本是训练时间翻倍。

---
方向六：训练稳定性加固（投入小）

Fold 3 和独立验证集虽然结果接近，但在不同 seed 下仍会波动。建议一个低成本改进：

```python
# ts_cross_val.py 中每个 fold 训练 3 次，用预测排名的平均值
for seed in [42, 123, 777]:
    set_seed(seed)
    train_model(...)
    df_preds_seed = generate_predictions(...)
    all_pred_ranks.append(df_preds_seed.groupby('trade_date')['pred'].rank(pct=True))

# 取三个 seed 的平均排名
df_preds['avg_rank'] = sum(all_pred_ranks) / 3
df_preds = df_preds.sort_values('avg_rank', ascending=False)
```

改动很小（加一个 seed 参数和循环），能显著降低单次初始化的运气成分。

---
优先级排序

┌────────┬─────────────────┬──────────────────┬──────────┐
│ 优先级 │      方向       │     预期效果     │  改动量  │
├────────┼─────────────────┼──────────────────┼──────────┤
│ P0     │ IC 评估对齐     │ 揭示真实模型能力 │ 一行代码 │
├────────┼─────────────────┼──────────────────┼──────────┤
│ P0     │ 截面特征        │ IC 提升 30-50%   │ ~20 行   │
├────────┼─────────────────┼──────────────────┼──────────┤
│ P1     │ 训练稳定性加固  │ 降低 seed 方差   │ ~15 行   │
├────────┼─────────────────┼──────────────────┼──────────┤
│ P1     │ 宏观择时升级    │ 压降 Max DD      │ ~30 行   │
├────────┼─────────────────┼──────────────────┼──────────┤
│ P2     │ 标签非重叠/周频 │ 可能提升稳定性   │ 改动较大 │
├────────┼─────────────────┼──────────────────┼──────────┤
│ P2     │ 分层建模        │ 进一步改善 IC    │ 改动较大 │
└────────┴─────────────────┴──────────────────┴──────────┘

建议从 P0 两项同时开始：IC
对齐让你看到真实的模型能力，截面特征直接补充模型最缺的信息维度。两者改动量极小、无风险、效果确定。