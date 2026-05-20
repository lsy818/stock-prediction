# V4.2 策略目标与损失优化计划 (T+5 收益与混合排序损失)

为了降低收益率预测的噪声并使模型损失函数更贴合选股目标，我们提议：
1. **数据端**：在 `data_preprocessing.py` 中增加 $T+5$ 累计收益率 `label_return_5d`。
2. **训练端**：在 `train.py` 中引入 **Hybrid Loss (MSE + Pairwise Margin Loss)** 混合排序损失函数，并通过 `loss_type` 和 `target_col` 参数支持灵活配置。
3. **回测端**：配合前述双阈值缓冲带机制，进行全面的 3-Fold TSCV 评估。

## User Review Required

> [!IMPORTANT]
> 1. **5日收益率标签 (`label_return_5d`) 的引入**：
>    * 我们将对数据预处理逻辑进行升级，新增 `label_return_5d`（未来 5 个交易日的累计收益率之和，可平摊日内噪声）。
>    * 在特征提取时，我们将同时保留 `label_return_1d` 和 `label_return_5d`，确保回测与训练的自由切换。
> 2. **混合排序损失 (`Hybrid Loss`) 的设计**：
>    * 计算公式为：$\text{Loss} = \alpha \cdot \text{MSE} + (1 - \alpha) \cdot \text{Pairwise Margin Loss}$。
>    * 在每个 Batch (8192 个样本) 中，我们按交易日对样本进行分组，在同一交易日内的股票之间随机两两配对，以计算排序梯度。这对于提升 Rank IC 有着极其直接的作用。

## Proposed Changes

### 1. 数据预处理层 (`data_preprocessing.py`)

#### [MODIFY] [data_preprocessing.py](file:///d:/日常/大学用/大三下/stock-testing/test/V4/data_preprocessing.py)
* 在 `generate_labels` 函数中，同时计算并生成 `label_return_1d` 与 `label_return_5d` 两个标签列：
  ```python
  df['label_return_1d'] = df.groupby('ts_code')['pct_chg'].shift(-1)
  df['label_return_5d'] = df.groupby('ts_code')['pct_chg'].transform(lambda x: x.shift(-1).rolling(5).sum().shift(-4))
  ```
* 保证在删除 NaN 时，不会因新标签列的加入导致样本数量异常缩水。

---

### 2. 数据集加载层 (`dataset.py`)

#### [MODIFY] [dataset.py](file:///d:/日常/大学用/大三下/stock-testing/test/V4/dataset.py)
* 修改 `get_dataloaders` 的参数，允许通过 `target_col`（默认为 `'label_return_1d'`）指定目标标签。
* **安全性防泄露修改**：在构建 `feature_cols` 时，显式地将 `['label_return_1d', 'label_return_5d']` 全部从特征列中剔除，防止模型把未使用的标签作为特征导致数据泄露。

---

### 3. 模型训练层 (`train.py` 与 `ts_cross_val.py`)

#### [MODIFY] [train.py](file:///d:/日常/大学用/大三下/stock-testing/test/V4/train.py)
* **实现 `hybrid_loss` 混合损失计算函数**（按交易日分组随机两两配对，计算 Margin Ranking Loss 与 MSE 的加权和）。
* 在 `train_model` 函数中增加 `target_col` 与 `loss_type` 参数，支持 MSELoss 和 Hybrid Loss 的切换。
* 更新 `train_one_epoch` 与 `validate`，在计算损失时将 `dates` 传入。

#### [MODIFY] [ts_cross_val.py](file:///d:/日常/大学用/大三下/stock-testing/test/V4/ts_cross_val.py)
* 在主流程中支持使用新参数配置进行 3-Fold TSCV。

---

## Verification Plan

### Automated Tests
1. **数据生成**：运行 `python data_preprocessing.py` 重新生成 parquet 数据集。
2. **混合损失与T+5收益训练**：在 `ts_cross_val.py` 中配置 `target_col='label_return_5d'` 且 `loss_type='hybrid'` 运行 3-Fold 滚动交叉验证，记录并对比新的回测绩效指标。
