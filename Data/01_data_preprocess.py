#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块
功能：加载和清洗原始拓扑数据和日志数据，构建图结构
输入：train和test文件夹中的.log.topo.json文件
输出：处理后的图数据结构，保存到data文件夹
"""

import json
import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.output_dir = os.path.join(data_dir, 'data')

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def load_topo_data(self, file_path):
        """
        加载拓扑数据
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def load_rootcause_data(self, file_path):
        """
        加载根因标签数据（仅训练集）
        """
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_node_features(self, node):
        """
        从节点中提取特征
        特征包括：节点类别、入度、出度、供应商等
        """
        features = {}
        features['class'] = node.get('@class', 'Unknown')
        features['rid'] = node.get('@rid', '')
        features['label'] = node.get('zh_label', '')
        features['in_degree'] = len(node.get('in_dependOn', []))
        features['out_degree'] = len(node.get('out_dependOn', []))
        features['vendor'] = node.get('vendor_name', 'Unknown')
        features['device_type'] = node.get('device_type', 'Unknown')

        return features

    def build_graph_structure(self, topo_data):
        """
        构建图结构
        返回：节点字典、边列表、节点特征
        """
        nodes = {}
        edges = []
        node_features = []

        # 处理节点
        for node in topo_data.get('nodes', []):
            rid = node.get('@rid')
            if rid:
                nodes[rid] = node
                features = self.extract_node_features(node)
                node_features.append(features)

                # 提取出边
                out_deps = node.get('out_dependOn', [])
                for target in out_deps:
                    edges.append((rid, target))

        return nodes, edges, node_features

    def extract_rootcause_labels(self, rootcause_data):
        """
        提取根因标签
        """
        if not rootcause_data:
            return []

        labels = []
        for rc in rootcause_data.get('rootcause', []):
            labels.append({
                'rid': rc.get('@rid'),
                'title': rc.get('title'),
                'location': rc.get('location'),
                'reason': rc.get('reason')
            })
        return labels

    def process_dataset(self, data_dir, is_train=True):
        """
        处理数据集（训练集或测试集）
        """
        processed_data = []

        # 获取所有子文件夹
        case_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        print(f"处理{'训练' if is_train else '测试'}数据集，共 {len(case_dirs)} 个案例")

        for case_id in tqdm(case_dirs):
            case_path = os.path.join(data_dir, case_id)
            topo_file = os.path.join(case_path, f"{case_id}.log.topo.json")
            rootcause_file = os.path.join(case_path, f"{case_id}.rootcause.json")

            # 加载拓扑数据
            if not os.path.exists(topo_file):
                continue

            topo_data = self.load_topo_data(topo_file)
            nodes, edges, node_features = self.build_graph_structure(topo_data)

            # 加载根因标签（仅训练集）
            labels = None
            if is_train and os.path.exists(rootcause_file):
                rootcause_data = self.load_rootcause_data(rootcause_file)
                labels = self.extract_rootcause_labels(rootcause_data)

            processed_data.append({
                'case_id': case_id,
                'timestamp': topo_data.get('time'),
                'nodes': nodes,
                'edges': edges,
                'node_features': node_features,
                'labels': labels
            })

        return processed_data

    def compute_statistics(self, train_data, test_data):
        """
        计算数据集统计信息
        """
        stats = {
            'train_cases': len(train_data),
            'test_cases': len(test_data),
            'node_classes': set(),
            'vendors': set(),
            'device_types': set()
        }

        for data in train_data + test_data:
            for features in data['node_features']:
                stats['node_classes'].add(features['class'])
                stats['vendors'].add(features['vendor'])
                stats['device_types'].add(features['device_type'])

        # 转换set为list以便保存
        stats['node_classes'] = list(stats['node_classes'])
        stats['vendors'] = list(stats['vendors'])
        stats['device_types'] = list(stats['device_types'])

        return stats

    def run(self):
        """
        执行完整的数据预处理流程
        """
        print("=" * 50)
        print("开始数据预处理")
        print("=" * 50)

        # 处理训练集
        train_data = self.process_dataset(self.train_dir, is_train=True)

        # 处理测试集
        test_data = self.process_dataset(self.test_dir, is_train=False)

        # 计算统计信息
        stats = self.compute_statistics(train_data, test_data)

        # 保存处理后的数据
        print("\n保存处理后的数据...")
        with open(os.path.join(self.output_dir, 'train_data.pkl'), 'wb') as f:
            pickle.dump(train_data, f)

        with open(os.path.join(self.output_dir, 'test_data.pkl'), 'wb') as f:
            pickle.dump(test_data, f)

        with open(os.path.join(self.output_dir, 'stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"\n数据预处理完成！")
        print(f"训练集: {stats['train_cases']} 个案例")
        print(f"测试集: {stats['test_cases']} 个案例")
        print(f"节点类别数: {len(stats['node_classes'])}")
        print(f"供应商数: {len(stats['vendors'])}")
        print(f"设备类型数: {len(stats['device_types'])}")
        print("=" * 50)


if __name__ == '__main__':
    # 设置数据目录（根据实际情况修改）
    preprocessor = DataPreprocessor(data_dir='./')
    preprocessor.run()
