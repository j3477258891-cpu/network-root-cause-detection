# 网络故障根因分析系统

## 项目概述
本项目是一个基于图神经网络和机器学习的网络故障根因分析系统，通过分析网络拓扑结构和设备日志，自动识别故障的根本原因。

## 目录结构
```
./
├── train/                          # 训练数据集
├── test/                           # 测试数据集
├── 01_data_preprocess.py          # 数据预处理模块
├── 02_feature_extraction.py       # 特征提取模块
├── 03_model_training.py           # 模型训练模块
├── 04_model_prediction.py         # 模型预测模块
├── 05_format_results.py           # 结果格式化模块
├── run_all.py                     # 主运行脚本
├── requirements.txt               # Python依赖包
├── install_dependencies.sh        # 依赖安装脚本
├── data/                          # 预处理后的数据（自动生成）
├── features/                      # 提取的特征（自动生成）
├── models/                        # 训练的模型（自动生成）
├── results/                       # 预测结果（自动生成）
└── submit/                        # 最终提交文件（自动生成）
```

## 快速开始

### 1. 安装依赖
```bash
bash install_dependencies.sh
```

或者手动安装：
```bash
pip install -r requirements.txt
```

### 2. 运行完整流程
```bash
python run_all.py
```

这将依次执行以下步骤：
1. 数据预处理
2. 特征提取
3. 模型训练
4. 模型预测
5. 结果格式化

### 3. 分步执行（可选）
如果需要分步执行或调试，可以按顺序运行：

```bash
# 步骤1: 数据预处理
python 01_data_preprocess.py

# 步骤2: 特征提取
python 02_feature_extraction.py

# 步骤3: 模型训练
python 03_model_training.py

# 步骤4: 模型预测
python 04_model_prediction.py

# 步骤5: 结果格式化
python 05_format_results.py
```

## 提交文件
运行完成后，提交文件位于 `./submit/` 目录下，每个测试案例生成一个对应的 `.rootcause.json` 文件。

## 系统要求
- Python 3.7+
- CPU: 8核心以上推荐
- 内存: 16GB以上推荐
- 磁盘空间: 10GB以上

## 预估运行时间
- 数据预处理: 约30分钟
- 特征提取: 约2小时
- 模型训练: 约1小时
- 模型预测: 约30分钟
- 结果格式化: 约10分钟
- **总计: 约4-5小时**（具体时间取决于硬件配置）
