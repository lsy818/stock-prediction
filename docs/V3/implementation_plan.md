# V3 全市场标的池与实盘交易限制扩展方案

为了进一步挖掘 Transformer 模型在横向海量数据上的泛化能力，我们计划将标的池从沪深 300 扩展至全市场 A 股（剔除北交所及 ST 股票），并在回测引擎中引入严格的涨跌停无法成交限制，使其无限逼近真实实盘环境。

## User Review Required
> [!IMPORTANT]  
> 1. 数据量将从 500 多只股票飙升至近 4500 只，总体量扩大约 8 倍。目前由于 Transformer 显存消耗仅约 2.7GB，我们计划将 `batch_size` 翻倍至 `8192` 以加快训练速度。
> 2. 由于全量数据的预处理和模型训练将会非常耗时（预计预处理需要 10+ 分钟，训练可能需要 30+ 分钟），建议耐心等待。是否同意将 `epochs` 先设置为 `15` 并保持 `patience=5` 的早停机制以节约时间？

## 提出的修改项

### 1. 迁移代码至 `test/V3`
- 将 `test/V2` 中的 `dataset.py`, `model.py`, `train.py`, `backtest.py`, `data_preprocessing.py` 等核心脚本全部拷贝到新建的 `test/V3` 文件夹中，以保证历史版本隔离。

### 2. [MODIFY] `test/V3/data_preprocessing.py`
重构数据拉取逻辑，从全市场获取数据：
- **移除 CSI300 限制**：将 `get_csi300_symbols()` 改为 `get_all_symbols()`。
- **北交所剔除**：在提取符号时过滤掉所有以 `.BJ` 结尾的股票。
- **ST 剔除**：保留并沿用先前的 ST 股票剔除逻辑，彻底筛除历史带 ST 标识的劣质资产。
- **保存路径**：新生成的全市场特征文件保存为 `../../data/processed/all_stocks_features.parquet`。

### 3. [MODIFY] `test/V3/backtest.py`
引入涨跌停撮合限制与路径更新：
- **数据路径更新**：将读取的 parquet 文件修改为 `all_stocks_features.parquet`。
- **加入 `pre_close`**：在 `df_raw_indexed` 中额外加载 `pre_close` 字段，为计算涨跌停价提供基准。
- **涨跌停判定逻辑**：
  - **主板** (00, 60开头)：10% 涨跌幅限制 (`Limit Up = round(pre_close * 1.10, 2)`)。
  - **创业板/科创板** (300, 688开头)：20% 涨跌幅限制 (`Limit Up = round(pre_close * 1.20, 2)`)。
- **撮合拦截机制**：
  - 卖出拦截：若 $T+1$ 开盘价等于或低于跌停价，判定为一字跌停，该股票**拒绝卖出**，原仓位强制保留，现金流不增加。
  - 买入拦截：若 $T+1$ 开盘价等于或高于涨停价，判定为一字涨停，该股票**拒绝买入**，目标仓位强制作废，现金流保留。

### 4. [MODIFY] `test/V3/train.py`
- **数据路径更新**：指定至 `../../data/processed/all_stocks_features.parquet`。
- **参数适配**：为加速大数据量训练，调整 `batch_size = 8192`。

## 验证计划
1. 执行 `data_preprocessing.py`，确认输出的数据集包含所有非 BJ、非 ST 的全市场股票（预计四千余只）。
2. 执行 `train.py`，监控 GPU 显存占用，确保显存稳定在 5~7GB。
3. 执行 `backtest.py`，并在终端监控“由于跌停无法卖出”或“由于涨停无法买入”的拦截日志，最终校验回测收益。
