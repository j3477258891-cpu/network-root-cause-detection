#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型预测模块
功能：使用训练好的模型对测试集进行预测
输入：features文件夹中的测试集特征，models文件夹中的训练模型
输出：预测结果，保存到results文件夹
"""

import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class RootCausePredictor:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.features_dir = os.path.join(data_dir, 'features')
        self.models_dir = os.path.join(data_dir, 'models')
        self.results_dir = os.path.join(data_dir, 'results')

        os.makedirs(self.results_dir, exist_ok=True)

        # 加载模型和标准化器
        self.load_models()

    def load_models(self):
        """
        加载训练好的模型
        """
        print("加载模型...")
        model_file = os.path.join(self.models_dir, 'rootcause_model.pkl')
        scaler_file = os.path.join(self.models_dir, 'scaler.pkl')

        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)

        print("模型加载完成！")

    def feature_dict_to_vector(self, feat_dict):
        """
        将特征字典转换为向量
        """
        feature_values = []
        for key in sorted(feat_dict.keys()):
            value = feat_dict[key]
            if isinstance(value, (int, float)):
                feature_values.append(value)
            else:
                feature_values.append(0)

        return np.array(feature_values, dtype=np.float32)

    def predict_case(self, case_data):
        """
        预测单个案例的根因节点
        """
        node_ids = case_data['node_ids']
        node_features = case_data['node_features']
        graph_features = case_data['graph_features']

        # 构建特征矩阵
        X = []
        for node_feat in node_features:
            feat_vector = self.feature_dict_to_vector(node_feat)
            graph_feat_vector = self.feature_dict_to_vector(graph_features)
            combined_feat = np.concatenate([feat_vector, graph_feat_vector])
            X.append(combined_feat)

        X = np.array(X)

        # 标准化
        X_scaled = self.scaler.transform(X)

        # 模型预测
        rf_prob = self.model['rf'].predict_proba(X_scaled)[:, 1]
        gb_prob = self.model['gb'].predict_proba(X_scaled)[:, 1]

        # 集成预测（概率平均）
        ensemble_prob = (rf_prob + gb_prob) / 2

        # 选择Top-K个最可能的根因节点
        # 根据实际情况调整K值
        top_k = min(5, max(1, int(len(node_ids) * 0.01)))  # 至少1个，至多5个，或1%的节点数
        top_k_indices = np.argsort(ensemble_prob)[-top_k:][::-1]

        # 构建预测结果
        predictions = []
        for idx in top_k_indices:
            if ensemble_prob[idx] > 0.3:  # 设置阈值过滤低概率预测
                predictions.append({
                    'node_id': node_ids[idx],
                    'probability': float(ensemble_prob[idx])
                })

        return predictions

    def predict_test_set(self):
        """
        对整个测试集进行预测
        """
        print("\n开始预测测试集...")

        # 加载测试集特征
        test_features_file = os.path.join(self.features_dir, 'test_features.pkl')
        with open(test_features_file, 'rb') as f:
            test_features = pickle.load(f)

        print(f"测试集案例数: {len(test_features)}")

        # 对每个案例进行预测
        all_predictions = {}
        for case_data in tqdm(test_features):
            case_id = case_data['case_id']
            predictions = self.predict_case(case_data)
            all_predictions[case_id] = predictions

        return all_predictions

    def save_predictions(self, predictions):
        """
        保存预测结果
        """
        # 保存为pickle格式
        pred_file = os.path.join(self.results_dir, 'predictions.pkl')
        with open(pred_file, 'wb') as f:
            pickle.dump(predictions, f)

        # 保存为JSON格式（便于查看）
        pred_json_file = os.path.join(self.results_dir, 'predictions.json')
        with open(pred_json_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        print(f"\n预测结果已保存到: {self.results_dir}")

    def run(self):
        """
        执行完整的预测流程
        """
        print("=" * 50)
        print("开始根因预测")
        print("=" * 50)

        # 预测测试集
        predictions = self.predict_test_set()

        # 保存预测结果
        self.save_predictions(predictions)

        # 统计信息
        total_cases = len(predictions)
        cases_with_predictions = sum(1 for p in predictions.values() if len(p) > 0)

        print("\n预测统计:")
        print(f"总案例数: {total_cases}")
        print(f"有预测结果的案例数: {cases_with_predictions}")
        print(f"平均每个案例预测根因数: {sum(len(p) for p in predictions.values()) / total_cases:.2f}")

        print("\n预测完成！")
        print("=" * 50)


if __name__ == '__main__':
    predictor = RootCausePredictor(data_dir='./')
    predictor.run()
