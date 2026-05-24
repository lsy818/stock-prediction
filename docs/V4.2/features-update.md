## 引入截面特征（投入小、效果确定）

当前模型只看个股自身的历史序列，完全不知道同日其他股票的表现。股市的核心 alpha 来源之一是相对强弱——一只股票涨 2%，如果板块涨 5%，它其实在走弱。

考虑新增特征：

| 特征 | 计算方式 | 含义 | 备注 | 
| ---- | ------- | ---- | ---- |
| cross_rank_pct_chg │ 当日该股涨跌幅在全市场的分位数 (0~1) │ 截面相对强弱 | |
│ cross_rank_vol │ 当日成交量分位数 │ 是否为市场焦点 | 大盘股成交量天然高，考虑用换手率代替 | 
| cross_rank_turnover_rate | | 换手率分位数 | 换手率是归一化的，跨股票可比 |
│ cross_rank_mom5 │ 5 日动量分位数 │ 中期相对强弱 | |
│ industry_rel_return │ 个股收益 − 同行业平均收益 │ 行业内 alpha | | 
│ industry_mom │ 同行业近 5 日平均收益 │ 板块动量 | |
| volume_ratio | 当日量 / 5 日均量 | 量比 | 已经是截面可比的相对指标 |


实现方式是事后填充——data_preprocessing.py 中按日期 groupby 计算截面 rank 即可，不引入 look-ahead。这几个特征加到现有的 ~50 个特征中，几乎零额外成本，但提供了模型当前完全缺失的信息维度。

### 不足之处

1. 行业分类的粒度和稳定性未知

basic.csv 的 industry 字段来自 Tushare，可能是申万行业分类（约 28 个大类）或证监会分类（约 19 个）。800 只股票分布在 20~30 个行业中，平均每行业约 30 只。但实际分布并不均匀——制造业可能占一半以上的股票。

如果 industry 粒度太粗（比如"制造业"涵盖 400+ 只股票），industry_rel_return（个股 - 行业均值）几乎等于"个股 - 全市场均值"，和 cross_rank_pct_chg 高度共线，5 个特征中至少有 2 个是冗余的。

可能需要先跑一次 basic.csv 的 industry 值分布统计，确认分类粒度。

2. "不引入 look-ahead"的论断需要验证

cross_rank_mom5 基于 mom5 = close.pct_change(5)。mom5 在时间点 T 是已知的（用了 T-4 到 T 的收盘价）。但 data_preprocessing.py 中计算 mom5 是在全部数据 concat 后做的，此时未来数据已经存在。只要 groupby('ts_code').transform(pct_change(5)) 只用到当前行及之前的数据，就是安全的——pct_change(5) 确实只向前看 5 行，且 pandas 的 transform 在 groupby 内是按时间排序处理的，不会泄露未来。

实际上真正需要验证的是：data_preprocessing.py:96 的 ffill() 先做了前向填充，然后才计算技术指标。如果某只股票在 T 日停牌、T+1 日复牌，T 日的缺失值被 T-1 日数据填充。这是安全的（用了过去数据）。

3. 缺少对已有高信息量特征的截面化

当前代码（data_preprocessing.py:143-146）已经从 moneyflow/ 构造了：

- df['net_mf_amount_ma5']
- df['mf_to_amount_ratio']

这些资金流特征是当前模型可能依赖的重要信号。将它们的截面排名加入特征集会比 cross_rank_vol 更有价值：
- cross_rank_net_mf_amount_ma5：资金净流入的截面相对强弱
- cross_rank_mf_to_amount_ratio：资金流向强度的截面比较

目前规划没有考虑利用 moneyflow/ 数据做截面化，浪费了数据中已有的高质量信息。

4. 特征共线性风险

新增 5 个特征后，pct_chg（原始涨跌幅）、mom5（5日动量）、cross_rank_pct_chg（涨跌幅排名）、cross_rank_mom5（动量排名）、industry_rel_return（行业内超额收益）这五个特征高度相关。在 15 天序列窗口中，GRU 可能难以区分哪些是独立信号。

可能需要在加入后跑一遍特征重要性分析（如 permutation importance），如果某些截面特征的贡献接近于零，则说明它与已有时间序列特征冗余。

5. 特征集建议

| #   | 特征名称 | 来源 | 作用 |
| --- |------- | ---- | ---- |
│ 1   │ cross_rank_pct_chg       │ daily/pct_chg           │ 当日截面相对强弱 │
│ 2   │ cross_rank_turnover_rate │ metric/turnover_rate    │ 换手活跃度截面排名 │
│ 3   │ cross_rank_volume_ratio  │ metric/volume_ratio     │ 量比异动截面排名（新增）│
│ 4   │ cross_rank_amount        │ daily/amount            │ 资金关注度截面排名（新增）│
│ 5   │ cross_rank_mom5          │ 衍生 mom5               │ 中期动量截面排名 │
│ 6   │ cross_rank_net_mf_ma5    │ 衍生 net_mf_amount_ma5  │ 资金净流入截面排名 │
│ 7   │ cross_rank_mf_ratio      │ 衍生 mf_to_amount_ratio │ 资金流向强度截面排名 │
│ 8   │ industry_rel_return      │ pct_chg + industry      │ 行业内超额收益 │
│ 9   │ industry_mom             │ mom5 + industry         │ 行业板块动量 │