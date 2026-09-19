#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练模块
功能：训练根因预测模型
输入：features文件夹中的特征数据
输出：训练好的模型，保存到models文件夹
"""

import json
import os
import pickle
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class RootCauseModel:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.features_dir = os.path.join(data_dir, 'features')
        self.models_dir = os.path.join(data_dir, 'models')

        os.makedirs(self.models_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.model = None

    def prepare_training_data(self, features_data):
        """
        准备训练数据
        将节点特征和标签对齐
        """
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

            # 为每个节点创建标签（故障节点标记为1，正常节点标记为0）
            for i, rid in enumerate(node_ids):
                # 转换特征字典为向量
                feat_vector = self.feature_dict_to_vector(node_features[i])

                # 添加图级别特征
                graph_feats = self.feature_dict_to_vector(case['graph_features'])
                feat_vector = np.concatenate([feat_vector, graph_feats])

                X.append(feat_vector)

                # 标签：是否为根因节点
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
        """
        将特征字典转换为向量
        """
        # 按照固定顺序提取特征值
        feature_values = []
        for key in sorted(feat_dict.keys()):
            value = feat_dict[key]
            if isinstance(value, (int, float)):
                feature_values.append(value)
            else:
                # 对于非数值特征，使用0填充
                feature_values.append(0)

        return np.array(feature_values, dtype=np.float32)

    def train_model(self, X_train, y_train):
        """
        训练模型
        使用集成学习方法：随机森林 + 梯度提升
        """
        print("\n开始训练模型...")

        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 处理类别不平衡：计算样本权重
        pos_weight = len(y_train) / (2 * np.sum(y_train))
        neg_weight = len(y_train) / (2 * (len(y_train) - np.sum(y_train)))
        sample_weights = np.where(y_train == 1, pos_weight, neg_weight)

        # 训练随机森林模型
        print("训练随机森林模型...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)

        # 训练梯度提升模型
        print("训练梯度提升模型...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        # 保存模型
        self.model = {
            'rf': rf_model,
            'gb': gb_model
        }

        print("模型训练完成！")

        return rf_model, gb_model

    def evaluate_model(self, X_val, y_val):
        """
        评估模型性能
        """
        X_val_scaled = self.scaler.transform(X_val)

        # 随机森林预测
        rf_pred = self.model['rf'].predict(X_val_scaled)
        rf_prob = self.model['rf'].predict_proba(X_val_scaled)[:, 1]

        # 梯度提升预测
        gb_pred = self.model['gb'].predict(X_val_scaled)
        gb_prob = self.model['gb'].predict_proba(X_val_scaled)[:, 1]

        # 集成预测（概率平均）
        ensemble_prob = (rf_prob + gb_prob) / 2
        ensemble_pred = (ensemble_prob > 0.5).astype(int)

        print("\n=== 随机森林模型评估 ===")
        print(f"准确率: {accuracy_score(y_val, rf_pred):.4f}")
        print(f"F1分数: {f1_score(y_val, rf_pred):.4f}")

        print("\n=== 梯度提升模型评估 ===")
        print(f"准确率: {accuracy_score(y_val, gb_pred):.4f}")
        print(f"F1分数: {f1_score(y_val, gb_pred):.4f}")

        print("\n=== 集成模型评估 ===")
        print(f"准确率: {accuracy_score(y_val, ensemble_pred):.4f}")
        print(f"F1分数: {f1_score(y_val, ensemble_pred):.4f}")

        return ensemble_prob

    def save_models(self):
        """
        保存训练好的模型
        """
        model_file = os.path.join(self.models_dir, 'rootcause_model.pkl')
        scaler_file = os.path.join(self.models_dir, 'scaler.pkl')

        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)

        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"\n模型已保存到: {self.models_dir}")

    def run(self):
        """
        执行完整的训练流程
        """
        print("=" * 50)
        print("开始模型训练")
        print("=" * 50)

        # 加载特征数据
        print("\n加载训练数据...")
        train_features_file = os.path.join(self.features_dir, 'train_features.pkl')
        with open(train_features_file, 'rb') as f:
            train_features = pickle.load(f)

        # 准备训练数据
        X, y, case_indices = self.prepare_training_data(train_features)

        # 划分训练集和验证集（按案例划分，避免数据泄露）
        unique_cases = np.unique(case_indices)
        np.random.seed(42)
        np.random.shuffle(unique_cases)

        split_idx = int(0.8 * len(unique_cases))
        train_cases = unique_cases[:split_idx]
        val_cases = unique_cases[split_idx:]

        train_mask = np.isin(case_indices, train_cases)
        val_mask = np.isin(case_indices, val_cases)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        print(f"\n训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")

        # 训练模型
        self.train_model(X_train, y_train)

        # 评估模型
        print("\n在验证集上评估模型...")
        self.evaluate_model(X_val, y_val)

        # 保存模型
        self.save_models()

        print("\n模型训练完成！")
        print("=" * 50)


if __name__ == '__main__':
    trainer = RootCauseModel(data_dir='./')
    trainer.run()
