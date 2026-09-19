#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版模型训练模块
功能：使用XGBoost、LightGBM和深度集成学习
输入：features文件夹中的增强特征数据
输出：训练好的模型，保存到models文件夹
"""

import json
import os
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("警告: XGBoost未安装，将使用其他模型")

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False
    print("警告: LightGBM未安装，将使用其他模型")


class EnhancedRootCauseModel:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.features_dir = os.path.join(data_dir, 'features')
        self.models_dir = os.path.join(data_dir, 'models')

        os.makedirs(self.models_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.models = {}

    def prepare_training_data(self, features_data):
        """准备训练数据"""
        X = []
        y = []
        case_indices = []

        for case_idx, case in enumerate(features_data):
            node_ids = case['node_ids']
            node_features = case['node_features']
            labels = case.get('labels', [])

            if not labels:
                continue

            # 提取标签中的rid
            label_rids = set([label['rid'] for label in labels])

            # 为每个节点创建标签
            for i, rid in enumerate(node_ids):
                # 转换特征字典为向量
                feat_vector = self.feature_dict_to_vector(node_features[i])

                # 添加图级别特征
                graph_feats = self.feature_dict_to_vector(case['graph_features'])
                feat_vector = np.concatenate([feat_vector, graph_feats])

                X.append(feat_vector)
                y.append(1 if rid in label_rids else 0)
                case_indices.append(case_idx)

        X = np.array(X)
        y = np.array(y)
        case_indices = np.array(case_indices)

        print(f"训练样本数: {len(X)}")
        print(f"正样本数: {np.sum(y == 1)}, 负样本数: {np.sum(y == 0)}")
        print(f"正负样本比例: 1:{np.sum(y == 0) / max(np.sum(y == 1), 1):.2f}")

        return X, y, case_indices

    def feature_dict_to_vector(self, feat_dict):
        """将特征字典转换为向量"""
        feature_values = []
        for key in sorted(feat_dict.keys()):
            value = feat_dict[key]
            if isinstance(value, (int, float)):
                feature_values.append(value)
            else:
                feature_values.append(0)
        return np.array(feature_values, dtype=np.float32)

    def train_models(self, X_train, y_train):
        """训练多个模型"""
        print("\n开始训练模型...")

        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 计算样本权重
        pos_weight = len(y_train) / (2 * max(np.sum(y_train), 1))
        neg_weight = len(y_train) / (2 * max(len(y_train) - np.sum(y_train), 1))
        sample_weights = np.where(y_train == 1, pos_weight, neg_weight)

        # 1. 随机森林
        print("训练随机森林...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['rf'] = rf_model

        # 2. 梯度提升
        print("训练梯度提升...")
        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=12,
            learning_rate=0.03,
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        self.models['gb'] = gb_model

        # 3. XGBoost
        if HAS_XGB:
            print("训练XGBoost...")
            scale_pos_weight = (len(y_train) - np.sum(y_train)) / max(np.sum(y_train), 1)
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_scaled, y_train)
            self.models['xgb'] = xgb_model

        # 4. LightGBM
        if HAS_LGB:
            print("训练LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train_scaled, y_train)
            self.models['lgb'] = lgb_model

        print(f"模型训练完成！共训练了 {len(self.models)} 个模型")

    def evaluate_models(self, X_val, y_val):
        """评估模型性能"""
        X_val_scaled = self.scaler.transform(X_val)

        predictions = {}
        probabilities = {}

        # 评估每个模型
        for name, model in self.models.items():
            pred = model.predict(X_val_scaled)
            prob = model.predict_proba(X_val_scaled)[:, 1]

            predictions[name] = pred
            probabilities[name] = prob

            print(f"\n=== {name.upper()} 模型评估 ===")
            print(f"准确率: {accuracy_score(y_val, pred):.4f}")
            print(f"精确率: {precision_score(y_val, pred, zero_division=0):.4f}")
            print(f"召回率: {recall_score(y_val, pred, zero_division=0):.4f}")
            print(f"F1分数: {f1_score(y_val, pred, zero_division=0):.4f}")

        # 集成预测（加权平均）
        # XGBoost和LightGBM权重更高
        weights = {
            'rf': 0.15,
            'gb': 0.15,
            'xgb': 0.35,
            'lgb': 0.35
        }

        # 根据实际可用模型调整权重
        available_models = list(self.models.keys())
        total_weight = sum(weights.get(m, 0.25) for m in available_models)
        normalized_weights = {m: weights.get(m, 0.25) / total_weight for m in available_models}

        ensemble_prob = np.zeros_like(list(probabilities.values())[0])
        for name, prob in probabilities.items():
            ensemble_prob += prob * normalized_weights[name]

        ensemble_pred = (ensemble_prob > 0.5).astype(int)

        print(f"\n=== 集成模型评估 (权重: {normalized_weights}) ===")
        print(f"准确率: {accuracy_score(y_val, ensemble_pred):.4f}")
        print(f"精确率: {precision_score(y_val, ensemble_pred, zero_division=0):.4f}")
        print(f"召回率: {recall_score(y_val, ensemble_pred, zero_division=0):.4f}")
        print(f"F1分数: {f1_score(y_val, ensemble_pred, zero_division=0):.4f}")

        return ensemble_prob, normalized_weights

    def save_models(self, weights):
        """保存训练好的模型"""
        model_file = os.path.join(self.models_dir, 'rootcause_model_enhanced.pkl')
        scaler_file = os.path.join(self.models_dir, 'scaler_enhanced.pkl')
        weights_file = os.path.join(self.models_dir, 'model_weights.pkl')

        with open(model_file, 'wb') as f:
            pickle.dump(self.models, f)

        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(weights_file, 'wb') as f:
            pickle.dump(weights, f)

        print(f"\n模型已保存到: {self.models_dir}")

    def run(self):
        """执行完整的训练流程"""
        print("=" * 50)
        print("开始增强模型训练")
        print("=" * 50)

        # 加载特征数据
        print("\n加载训练数据...")
        train_features_file = os.path.join(self.features_dir, 'train_features_enhanced.pkl')
        with open(train_features_file, 'rb') as f:
            train_features = pickle.load(f)

        # 准备训练数据
        X, y, case_indices = self.prepare_training_data(train_features)

        # 划分训练集和验证集
        unique_cases = np.unique(case_indices)
        np.random.seed(42)
        np.random.shuffle(unique_cases)

        split_idx = int(0.85 * len(unique_cases))  # 增加训练集比例
        train_cases = unique_cases[:split_idx]
        val_cases = unique_cases[split_idx:]

        train_mask = np.isin(case_indices, train_cases)
        val_mask = np.isin(case_indices, val_cases)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        print(f"\n训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")

        # 训练模型
        self.train_models(X_train, y_train)

        # 评估模型
        print("\n在验证集上评估模型...")
        _, weights = self.evaluate_models(X_val, y_val)

        # 保存模型
        self.save_models(weights)

        print("\n模型训练完成！")
        print("=" * 50)


if __name__ == '__main__':
    trainer = EnhancedRootCauseModel(data_dir='./')
    trainer.run()
