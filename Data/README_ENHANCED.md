# 网络故障根因分析系统 - 增强版

## 优化目标
**提升F1分数从0.82到更高水平**

## 主要优化策略

### 1. 增强特征工程
- **语义特征**：从节点标签中提取设备类型、位置关键词
- **拓扑特征**：K-core值、邻居平均度数、二跳邻居数量
- **中心性特征**：改进的PageRank、Betweenness（大图采样计算）
- **度数比率**：入度/出度比率，反映节点依赖模式

### 2. 更强的模型
- **XGBoost**：梯度提升树，处理非线性关系能力强
- **LightGBM**：轻量级梯度提升，速度快且准确
- **Random Forest**：随机森林作为基准
- **Gradient Boosting**：传统梯度提升
- **加权集成**：XGBoost和LightGBM权重更高（各35%）

### 3. 智能Top-K选择策略
- **动态K值**：根据图规模自适应调整（1-8个）
- **概率阈值**：使用95分位数动态阈值
- **规则过滤**：根据节点类型调整概率
  - 提升：Equipment、Board、Power类节点（×1.2）
  - 降低：Service、Cell类节点（×0.8）

### 4. 改进的训练策略
- **样本权重**：解决类别不平衡问题
- **更大训练集**：85%训练 + 15%验证
- **超参数优化**：增加树的数量（300棵）、调整学习率（0.03-0.05）

### 5. 故障标题和原因映射
- **基于训练数据分析**：使用高频故障标题
- **智能匹配**：根据节点类别自动生成合理的故障描述

## 文件说明

### 核心代码（增强版）
- `00_analyze_training_data.py` - 训练数据深度分析
- `02_feature_extraction_enhanced.py` - 增强特征提取
- `03_model_training_enhanced.py` - 增强模型训练（XGBoost/LightGBM）
- `04_model_prediction_enhanced.py` - 增强预测（智能Top-K）
- `05_format_results_enhanced.py` - 结果格式化为CSV

### 通用代码
- `01_data_preprocess.py` - 数据预处理（通用）
- `run_all_enhanced.py` - 一键运行增强版流程

### 配置文件
- `requirements_enhanced.txt` - 增强版依赖包（含XGBoost/LightGBM）

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements_enhanced.txt
```

或者：
```bash
pip install numpy scipy scikit-learn networkx tqdm xgboost lightgbm
```

### 2. 运行完整流程
```bash
python run_all_enhanced.py
```

### 3. 分步运行（调试用）
```bash
# 可选：分析训练数据
python 00_analyze_training_data.py

# 步骤1：数据预处理
python 01_data_preprocess.py

# 步骤2：增强特征提取
python 02_feature_extraction_enhanced.py

# 步骤3：增强模型训练
python 03_model_training_enhanced.py

# 步骤4：增强预测
python 04_model_prediction_enhanced.py

# 步骤5：格式化为CSV
python 05_format_results_enhanced.py
```

### 4. 提交结果
运行完成后，在当前目录下会生成 `result_record.csv` 文件，直接提交到比赛平台即可。

## 预估运行时间
- 数据预处理: 30分钟
- 增强特征提取: 2.5小时（增加了更多特征计算）
- 增强模型训练: 1.5小时（多个模型训练）
- 增强预测: 40分钟
- 结果格式化: 10分钟
- **总计: 约5-6小时**

## 与基础版本的对比

| 特性 | 基础版本 | 增强版本 |
|------|---------|---------|
| 特征数量 | ~15个 | ~30个 |
| 模型数量 | 2个 (RF+GB) | 4个 (RF+GB+XGB+LGB) |
| Top-K策略 | 固定1-5个 | 动态1-8个 |
| 规则过滤 | 无 | 有 |
| 概率阈值 | 固定0.3 | 动态（95分位数）|
| 训练集比例 | 80% | 85% |
| 预期F1 | 0.82 | >0.85 |

## 进一步优化建议

如果F1分数仍不够高，可以尝试：

1. **深度学习模型**：使用Graph Neural Networks (GCN/GAT)
2. **时序特征**：利用timestamp提取时间相关特征
3. **文本特征**：对zh_label进行NLP处理
4. **集成学习**：尝试Stacking或更复杂的集成策略
5. **超参数调优**：使用Optuna或GridSearch优化超参数
6. **半监督学习**：利用测试集进行伪标签学习

## 注意事项

1. 确保Python版本 >= 3.7
2. 确保有足够的内存（推荐16GB+）
3. XGBoost和LightGBM会自动使用多核CPU
4. 如果没有GPU，模型仍能正常运行
5. 生成的CSV文件编码为UTF-8

## 疑难解答

### 如果XGBoost或LightGBM安装失败
```bash
# 使用conda安装（推荐）
conda install -c conda-forge xgboost lightgbm

# 或使用预编译轮子
pip install xgboost --prefer-binary
pip install lightgbm --prefer-binary
```

### 如果内存不足
- 减少特征提取时的样本数
- 减少模型中的树数量（n_estimators）
- 分批处理大图的中心性计算

### 如果运行时间过长
- 在大图上跳过Betweenness计算
- 减少模型数量（只使用XGBoost+LightGBM）
- 使用更少的树（n_estimators=200）

## 技术支持

遇到问题请检查：
1. 依赖包版本是否正确
2. 数据文件是否完整
3. 磁盘空间是否充足
4. Python环境是否正确

---

**祝比赛顺利！冲击更高分数！🚀**
