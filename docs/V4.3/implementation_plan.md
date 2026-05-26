# V4.2 截面特征升级计划 (Cross-sectional Features)

本计划旨在引入大量成本极低但信息量极高的截面相对强弱特征，补充现有时间序列模型的“上帝视角”。

## 发现与结论验证

1. **行业分类粒度极其精细**：
   我后台测试读取了 `basic.csv`，发现 `industry` 字段共有 **110 个独立类别**。在我们的 800 只候选股票池中，平均每个行业仅有约 7 只股票。这意味着 `industry_rel_return` 是极度微观的板块内阿尔法指标，不会退化为全市场 Rank，与 `cross_rank_pct_chg` 不会构成严重共线性。
2. **关于 Look-ahead 的安全性**：
   在截面特征计算中，我们将严格使用 `df.groupby('trade_date')['FEATURE'].rank(pct=True)`。由于 `groupby` 的是 `trade_date`，模型只会在当天（T日）的所有截面数据中打分，不会偷看 T+1 的数据。配合先 `ffill` 再计算衍生指标的逻辑，防泄漏完全闭环。
3. **资金流特征横截面化**：
   原始的 `net_mf_amount_ma5` 是绝对金额，受股票市值和市场整体热度影响极大，几乎不可比。加上横截面 Rank 后，就能立刻凸显出“哪些股票是当日的全市场资金焦点”。

## Proposed Changes

### 1. 数据预处理更新

#### [MODIFY] [data_preprocessing.py](file:///d:/日常/大学用/大三下/stock-testing/test/V4/data_preprocessing.py)
* **引入基础信息表**：读取 `basic.csv` 并提取 `ts_code` 和 `industry`，合并到每日行情表 `df` 中。
* **计算横截面 Rank 特征**：
  ```python
  def add_cross_sectional_features(df):
      print("Adding cross-sectional features...")
      
      # 1. 动量与量价 Rank
      df['cross_rank_pct_chg'] = df.groupby('trade_date')['pct_chg'].rank(pct=True)
      df['cross_rank_turnover_rate'] = df.groupby('trade_date')['turnover_rate'].rank(pct=True)
      df['cross_rank_mom5'] = df.groupby('trade_date')['mom5'].rank(pct=True)
      
      # 2. 资金流向 Rank
      df['cross_rank_net_mf_ma5'] = df.groupby('trade_date')['net_mf_amount_ma5'].rank(pct=True)
      df['cross_rank_mf_ratio'] = df.groupby('trade_date')['mf_to_amount_ratio'].rank(pct=True)
      
      # 3. 行业特征
      industry_mean_return = df.groupby(['trade_date', 'industry'])['pct_chg'].transform('mean')
      df['industry_rel_return'] = df['pct_chg'] - industry_mean_return
      df['industry_mom'] = df.groupby(['trade_date', 'industry'])['mom5'].transform('mean')
      
      # 缺失值处理：Rank 的中性值为 0.5，超额收益中性值为 0
      rank_cols = [c for c in df.columns if 'cross_rank' in c]
      df[rank_cols] = df[rank_cols].fillna(0.5)
      df['industry_rel_return'] = df['industry_rel_return'].fillna(0)
      df['industry_mom'] = df['industry_mom'].fillna(0)
      
      return df
  ```

### 2. 避免共线性风险
虽然深度学习中的注意力机制（Attention GRU）对于共线性特征（如 `pct_chg` vs `cross_rank_pct_chg`）具备极强的自适应降噪能力，但多余的参数可能会拖慢收敛。
我们将保持现有全量特征（将增加 7 个截面特征至总数 64 个），在跑完一轮 3-Fold 验证后，如果 IC 提升不明显，则启动 Permutation Importance 分析，剔除冗余项。

## Verification Plan

### Automated Tests
1. 运行 `python data_preprocessing.py` 重新生成带有横截面信息的 `800_stocks_features.parquet`。
2. 触发一次 `python ts_cross_val.py` 进行性能摸底。
3. 检查控制台输出，对比加上截面特征后的夏普是否有进一步提升。
