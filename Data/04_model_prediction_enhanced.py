#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版模型预测模块
功能：使用训练好的模型对测试集进行预测，采用智能Top-K策略
输入：features文件夹中的测试集特征，models文件夹中的训练模型
输出：预测结果，保存到results文件夹
"""

import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class EnhancedRootCausePredictor:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.features_dir = os.path.join(data_dir, 'features')
        self.models_dir = os.path.join(data_dir, 'models')
        self.results_dir = os.path.join(data_dir, 'results')

        os.makedirs(self.results_dir, exist_ok=True)

        # 加载模型和标准化器
        self.load_models()

    def load_models(self):
        """加载训练好的模型"""
        print("加载模型...")
        model_file = os.path.join(self.models_dir, 'rootcause_model_enhanced.pkl')
        scaler_file = os.path.join(self.models_dir, 'scaler_enhanced.pkl')
        weights_file = os.path.join(self.models_dir, 'model_weights.pkl')

        with open(model_file, 'rb') as f:
            self.models = pickle.load(f)

        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(weights_file, 'rb') as f:
            self.model_weights = pickle.load(f)

        print(f"模型加载完成！共加载 {len(self.models)} 个模型")
        print(f"模型权重: {self.model_weights}")

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

    def apply_rule_based_filtering(self, node_ids, nodes, probabilities):
        """
        应用基于规则的过滤
        根据节点类型和特征，调整预测概率
        """
        adjusted_probs = probabilities.copy()

        for i, rid in enumerate(node_ids):
            node = nodes.get(rid, {})
            node_class = node.get('@class', '').lower()
            label = node.get('zh_label', '').lower()

            # 提升特定类型节点的概率
            # 根因更可能是设备、单板、电源等
            if any(keyword in node_class for keyword in ['equipment', 'board', 'unit', 'power']):
                adjusted_probs[i] *= 1.2

            # 传输电路、基站也较常见
            if any(keyword in node_class for keyword in ['trans', 'circuit', 'basestation']):
                adjusted_probs[i] *= 1.1

            # 降低某些不太可能是根因的节点概率
            if any(keyword in node_class for keyword in ['service', 'cell', 'carrier']):
                adjusted_probs[i] *= 0.8

        return adjusted_probs

    def smart_topk_selection(self, probabilities, node_ids, nodes, graph_features):
        """
        智能Top-K选择策略
        根据图的规模、概率分布动态调整K值
        """
        num_nodes = graph_features.get('num_nodes', len(node_ids))

        # 基础K值：根据节点数量动态调整
        if num_nodes < 50:
            base_k = 1
        elif num_nodes < 100:
            base_k = 2
        elif num_nodes < 200:
            base_k = 3
        elif num_nodes < 500:
            base_k = 4
        else:
            base_k = 5

        # 根据概率分布调整K值
        # 如果有多个高概率节点，增加K值
        high_prob_count = np.sum(probabilities > 0.5)
        if high_prob_count > base_k:
            base_k = min(high_prob_count, 8)

        # 概率阈值：动态调整
        # 使用分位数方法
        if len(probabilities) > 0:
            threshold = max(0.3, np.percentile(probabilities, 95))
        else:
            threshold = 0.3

        # 选择Top-K个节点
        top_k_indices = np.argsort(probabilities)[-base_k:][::-1]

        # 过滤低于阈值的节点
        selected_indices = [idx for idx in top_k_indices if probabilities[idx] > threshold]

        # 确保至少有一个预测（如果所有概率都很低）
        if len(selected_indices) == 0 and len(top_k_indices) > 0:
            selected_indices = [top_k_indices[0]]

        return selected_indices

    def predict_case(self, case_data):
        """预测单个案例的根因节点"""
        node_ids = case_data['node_ids']
        node_features = case_data['node_features']
        graph_features = case_data['graph_features']
        nodes = case_data.get('nodes', {})

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

        # 集成预测
        ensemble_prob = np.zeros(len(X))
        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[:, 1]
            weight = self.model_weights.get(name, 0.25)
            ensemble_prob += prob * weight

        # 应用基于规则的过滤
        adjusted_prob = self.apply_rule_based_filtering(node_ids, nodes, ensemble_prob)

        # 智能Top-K选择
        selected_indices = self.smart_topk_selection(adjusted_prob, node_ids, nodes, graph_features)

        # 构建预测结果
        predictions = []
        for idx in selected_indices:
            predictions.append({
                'node_id': node_ids[idx],
                'probability': float(adjusted_prob[idx]),
                'node': nodes.get(node_ids[idx], {})
            })

        return predictions

    def predict_test_set(self):
        """对整个测试集进行预测"""
        print("\n开始预测测试集...")

        # 加载测试集特征
        test_features_file = os.path.join(self.features_dir, 'test_features_enhanced.pkl')
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
        """保存预测结果"""
        # 保存为pickle格式
        pred_file = os.path.join(self.results_dir, 'predictions_enhanced.pkl')
        with open(pred_file, 'wb') as f:
            pickle.dump(predictions, f)

        # 保存为JSON格式
        pred_json_file = os.path.join(self.results_dir, 'predictions_enhanced.json')
        # 移除不能序列化的node字段
        predictions_for_json = {}
        for case_id, preds in predictions.items():
            predictions_for_json[case_id] = [
                {'node_id': p['node_id'], 'probability': p['probability']}
                for p in preds
            ]

        with open(pred_json_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_for_json, f, ensure_ascii=False, indent=2)

        print(f"\n预测结果已保存到: {self.results_dir}")

    def run(self):
        """执行完整的预测流程"""
        print("=" * 50)
        print("开始增强根因预测")
        print("=" * 50)

        # 预测测试集
        predictions = self.predict_test_set()

        # 保存预测结果
        self.save_predictions(predictions)

        # 统计信息
        total_cases = len(predictions)
        cases_with_predictions = sum(1 for p in predictions.values() if len(p) > 0)
        avg_predictions = sum(len(p) for p in predictions.values()) / total_cases

        print("\n预测统计:")
        print(f"总案例数: {total_cases}")
        print(f"有预测结果的案例数: {cases_with_predictions}")
        print(f"平均每个案例预测根因数: {avg_predictions:.2f}")

        # 概率分布统计
        all_probs = [p['probability'] for preds in predictions.values() for p in preds]
        if all_probs:
            print(f"\n预测概率统计:")
            print(f"最小概率: {min(all_probs):.4f}")
            print(f"最大概率: {max(all_probs):.4f}")
            print(f"平均概率: {np.mean(all_probs):.4f}")
            print(f"中位数概率: {np.median(all_probs):.4f}")

        print("\n预测完成！")
        print("=" * 50)


if __name__ == '__main__':
    predictor = EnhancedRootCausePredictor(data_dir='./')
    predictor.run()
