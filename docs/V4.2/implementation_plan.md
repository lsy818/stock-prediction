# V4.2 策略目标与损失优化计划 (多日收益与混合排序损失)

为了降低收益率预测的噪声并使模型损失函数更贴合选股目标，我们提出以下升级计划。

## User Review Required

> [!IMPORTANT]
> 1. **特征防泄漏机制（极为关键）**：
>    在 `dataset.py` 中，显式地把 `['label_return_1d', 'label_return_3d', 'label_return_5d']` 全部从 `feature_cols` 中排除，彻底杜绝 Look-ahead Leakage。
> 2. **防 OOM 的全量配对采样机制**：
>    在同一批次中，提取相同日期的样本进行两两配对以最大化梯度利用率。为防止偶然情况下单日样本聚集导致组合数 $O(N^2)$ 爆炸引发 GPU OOM，我们引入 `MAX_PAIRS_PER_DAY = 1000`（或适当数值）。超出该阈值时进行随机采样截断。
> 3. **带真实收益差距 Mask 的 Soft Pairwise Loss**：
>    结合硬性筛选与平滑梯度的优势，使用 `mask = (torch.abs(target_i - target_j) > target_margin)`（如 `margin=0.001`，即收益率相差至少 0.1%）。
>    * **只对收益率差异明显的 pair 计算损失**，避免模型在微小噪声上过度拟合。
>    * 对被筛选出的 pair 使用 **Pairwise Logistic Loss** ($\log(1 + \exp(-y_{pair} (\hat{y}_i - \hat{y}_j)))$)，保证优异的梯度平滑性。
> 4. **多日预测与日频调仓的自洽性**：
>    预测 T+5 收益并用于日频调仓，在逻辑上完全自洽，且完美契合了上一版本引入的“双阈值缓冲带机制”（其实际拉长了单股持仓周期至数天）。

## Proposed Changes

### 1. 数据预处理层 (`data_preprocessing.py`)

#### [MODIFY] [data_preprocessing.py](file:///d:/日常/大学用/大三下/stock-testing/test/V4/data_preprocessing.py)
* 计算 1D, 3D, 5D 的前向收益率：
  ```python
  df['label_return_1d'] = df.groupby('ts_code')['pct_chg'].shift(-1)
  df['label_return_3d'] = df.groupby('ts_code')['pct_chg'].transform(lambda x: x.shift(-1).rolling(3).sum().shift(-2))
  df['label_return_5d'] = df.groupby('ts_code')['pct_chg'].transform(lambda x: x.shift(-1).rolling(5).sum().shift(-4))
  ```

---

### 2. 数据集加载层 (`dataset.py`)

#### [MODIFY] [dataset.py](file:///d:/日常/大学用/大三下/stock-testing/test/V4/dataset.py)
* 在生成特征数组前，强制剔除所有类型的标签：
  ```python
  label_cols = ['label_return_1d', 'label_return_3d', 'label_return_5d']
  feature_cols = [c for c in df.columns if c not in ['ts_code', 'trade_date'] + label_cols]
  ```
* 修改 `get_dataloaders` 函数签名，添加 `target_col='label_return_1d'` 默认参数供灵活切换。

---

### 3. 模型训练层 (`train.py` 与 `ts_cross_val.py`)

#### [MODIFY] [train.py](file:///d:/日常/大学用/大三下/stock-testing/test/V4/train.py)
* **编写带有 Mask 和防 OOM 的 Soft Pairwise Loss**：
  * 在同一天提取所有配对，应用 `MAX_PAIRS_PER_DAY` 截断。
  * 应用 `target_margin` 掩码进行过滤。
  * 计算有效 pair 的 Logistic Loss。
* **参数化混合权重**：修改 `train_model` 签名，添加 `alpha` 参数（默认为 0.5）。总损失等于 $\alpha \cdot \text{MSE} + (1 - \alpha) \cdot \text{SoftPairwiseLoss}$。

#### [MODIFY] [ts_cross_val.py](file:///d:/日常/大学用/大三下/stock-testing/test/V4/ts_cross_val.py)
* 使用 `target_col='label_return_5d'` 并在 `train_model` 中指定 `loss_type='hybrid'` 运行评估。

## Verification Plan

### Automated Tests
1. 运行 `python data_preprocessing.py` 更新数据集文件。
2. 运行 `python ts_cross_val.py` 启动交叉验证，并在控制台实时观察每个 Fold 训练时的 Loss 是否平稳下降。
