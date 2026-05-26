# V4.4 深度架构与数据流综合重构计划 (Updated)

本次重构结合了深刻的金融逻辑与工程实现，旨在彻底解决 V4 策略在训练中暴露的“Batch间方差剧烈震荡”、“梯度互相冲突”以及“网络内部协变量偏移”等深层问题。

## 方案讨论总结与采纳
1. **采用日内 Z-Score 替代 Rank 作为 Target**：极其精妙的提议！Z-Score 既统一了跨日 Target 方差（稳定 MSE 梯度），又保留了超额收益的相对幅度，且完美兼容了 `backtest.py` 中 `pred > 0`（即预期跑赢当日大盘均值）的择时底线逻辑。
2. **保留 `amount` 特征**：同意。`amount` 作为表征资金流水的绝对规模指标，对 GRU 识别市场流动性具有独特价值，剔除范围缩小至仅纯价格特征。
3. **保留 GRU Dropout**：同意。Dropout 负责层间正则化，与 LayerNorm 负责的特征分布稳定并不冲突，应共同发力。

## 确认通过后即将执行的更改

### 1. 数据预处理层 (Data Pipeline)

#### [MODIFY] `data_preprocessing.py`
- **目标值截面 Z-Score 化 (Target Normalization)**：
  在计算完 `label_return_5d` 后，追加计算：
  `df['norm_return_5d'] = df.groupby('trade_date')['label_return_5d'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))`
- **作用**：将跨日方差震荡巨大的绝对收益率转换为每日标准正态分布，彻底消灭 Batch 间 MSE 震荡的根源，并完美对齐回测买入逻辑。

#### [MODIFY] `dataset.py`
- **剔除绝对价格特征 (Feature Pruning)**：
  在 `feature_cols` 列表中，硬编码排除非平稳的绝对价格特征：`['open', 'high', 'low', 'close', 'pre_close', 'vwap']`。
- **作用**：掐断无量纲、带趋势的价格水平对 GRU 的干扰，强制模型学习动量、波动与截面相对强弱。

---

### 2. 模型架构层 (Architecture)

#### [MODIFY] `model.py`
- **引入 LayerNorm**：
  在 `AttentionGRU` 类的初始化中，保留原有的 `dropout` 参数。并在 GRU 后接一层 `nn.LayerNorm(hidden_size)`。
  在 `forward` 函数中，对 GRU 的输出序列执行 LayerNorm 后，再进入 Attention 层。
- **作用**：消除输入扰动带来的内部激活爆炸，防止出现跨日极端行情时的协变量偏移。

---

### 3. 训练与优化器层 (Training Loop)

#### [MODIFY] `train.py`
- **梯度累积 (Gradient Accumulation)**：
  在 `train_one_epoch` 中引入 `accumulation_steps = 4`。
  将 `scaler.scale(loss).backward()` 改为 `scaler.scale(loss / accumulation_steps).backward()`。
  在每 4 个 Batch 结束或 Dataloader 耗尽时，才执行 `scaler.unscale_`、`clip_grad_norm_` 和 `scaler.step`。
- **作用**：等价于将物理 Batch Size 从 20 天扩展到 80 天，极大平滑了由随机日期组合带来的梯度噪音。

#### [MODIFY] `ts_cross_val.py`
- **超参对齐**：
  将传入训练任务的 `target_col` 从 `'label_return_5d'` 更改为 `'norm_return_5d'`。
  保持 `alpha=0.5`（MSE 目标方差变为 1.0 后，MSE 梯度与 Pairwise 梯度的量级将自动达到可比较的平衡态，此参数先做观察）。

---

## Verification Plan

### Automated Tests
1. **数据生成**：执行 `data_preprocessing.py` 生成包含 `norm_return_5d` 的特征表。
2. **训练试跑**：执行 `ts_cross_val.py` 跑 1~2 个 Epoch，观察日志输出。

### 核心观察指标
1. **Batch MSE 极度稳定**：因为 Z-Score 的方差恒为 1，MSE Baseline 理论值约为 1.0。期待看到 Batch MSE 稳定在 0.8~1.0 的极小区间内。
2. **Pairwise Loss 破冰**：随着 MSE 不再野蛮干扰，期待观察到 Pairwise Loss 稳步下降，证明横截面排序开始发力。
3. **Pred Std 锚定**：预测方差应平稳增长并逐渐向 1.0 的理论标准差靠拢。
