### 1. 验证集、测试集问题

目前验证集存在以下问题：
1. 回测阶段完全忽略验证集：只回测2026年（测试集），导致验证集数据完全浪费
2. 缺少验证集策略评估：没有在验证集（2025年）上评估策略表现
3. 模型选择信息有限：只基于IC选择模型，缺少其他策略指标
4. 目前只给出了测试集的结果，而理论上需要使用验证集的结果调整参数等

存在的风险：
1. 过拟合：当前结构是 **训练(2016-2024) → 验证(2025，仅用于早停) → 测试(2026)**。模型可能在2025年验证集上已过拟合，但未检测到
2. 只测试了2026年1年的表现，不知道策略在不同市场环境下的表现、长期稳定性、参数敏感性

已经暴露的问题：**临时修改了代码，使用验证集 (2025.1-2025.12) 测试，得到的年化收益不足 20%。**

需要修复的问题：
1. 添加验证集回测选项
2. 考虑时间序列交叉验证，比如： 
```python 
train_periods = [
      ('2018-01-01', '2020-12-31'),  # 3年训练
      ('2019-01-01', '2021-12-31'),  # 滑动窗口
      ('2020-01-01', '2022-12-31'),
] 
# 每次用后续1年作为验证
val_period = 1  # 年
```
3. 是否需要延长验证集时间？
比如训练窗口：2016.1-2023.12，验证窗口：2024.1-2025.12


可能的优化方案：
```python
# 增加验证集回测
def comprehensive_backtest(model_path, data_path):
    """全面的回测：验证集 + 测试集"""

    # 1. 在验证集上回测（2025年）
    print("=== 验证集回测 (2025年) ===")
    val_preds = generate_predictions_on_validation(model_path, data_path)
    val_backtest_results = backtest(val_preds, val_raw_data, initial_cash=10000)

    # 2. 在测试集上回测（2026年）
    print("=== 测试集回测 (2026年) ===")
    test_preds = generate_predictions_on_test(model_path, data_path)
    test_backtest_results = backtest(test_preds, test_raw_data, initial_cash=10000)

    # 3. 对比分析
    compare_results(val_backtest_results, test_backtest_results)



# 完整的时间序列验证
def time_series_cross_validation(data_path, model_class, n_folds=3):
    """时间序列交叉验证"""
    results = []

    # 定义时间窗口
    time_windows = [
        ('2016-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),  # Fold 1
        ('2017-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),  # Fold 2
        ('2018-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),  # Fold 3
    ]

    for i, (train_start, train_end, val_start, val_end) in enumerate(time_windows):
        print(f"\n=== Fold {i+1}: 训练({train_start}-{train_end}) 验证({val_start}-{val_end}) ===")

        # 训练模型
        model = train_model_on_window(data_path, train_start, train_end, model_class)

        # 验证集回测
        val_results = backtest_on_window(model, data_path, val_start, val_end)
        results.append(val_results)

    # 统计结果
    analyze_cv_results(results)



# 修改backtest.py，添加验证集回测选项
def main():
    model_path = 'checkpoints/best_ensemble.pth'
    data_path = '../../data/processed/all_stocks_features.parquet'

    print("=== 阶段1: 验证集回测 (2025年) ===")
    df_preds_val = generate_predictions(model_path, data_path, dataset_type='val')
    df_raw_val = load_data_for_period(data_path, '2025-01-01', '2025-12-31')
    backtest(df_preds_val, df_raw_val, top_k=30, label="验证集")

    print("\n=== 阶段2: 测试集回测 (2026年) ===")
    df_preds_test = generate_predictions(model_path, data_path, dataset_type='test')
    df_raw_test = load_data_for_period(data_path, '2026-01-01', '2026-12-31')
    backtest(df_preds_test, df_raw_test, top_k=30, label="测试集")
```