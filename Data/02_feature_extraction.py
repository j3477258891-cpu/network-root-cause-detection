#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征提取模块
功能：从图结构中提取节点特征和图特征
输入：data文件夹中的处理后数据
输出：特征向量，保存到features文件夹
"""

import json
import os
import pickle
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import networkx as nx


class FeatureExtractor:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.data_folder = os.path.join(data_dir, 'data')
        self.output_dir = os.path.join(data_dir, 'features')

        os.makedirs(self.output_dir, exist_ok=True)

        # 加载统计信息
        with open(os.path.join(self.data_folder, 'stats.json'), 'r', encoding='utf-8') as f:
            self.stats = json.load(f)

        # 创建编码映射
        self.class_to_idx = {c: i for i, c in enumerate(self.stats['node_classes'])}
        self.vendor_to_idx = {v: i for i, v in enumerate(self.stats['vendors'])}
        self.device_to_idx = {d: i for i, d in enumerate(self.stats['device_types'])}

    def build_networkx_graph(self, nodes, edges):
        """
        构建NetworkX图用于计算图特征
        """
        G = nx.DiGraph()

        # 添加节点
        for rid, node in nodes.items():
            G.add_node(rid, **node)

        # 添加边
        for src, dst in edges:
            if src in G.nodes() and dst in G.nodes():
                G.add_edge(src, dst)

        return G

    def extract_node_centrality_features(self, G, rid):
        """
        提取节点中心性特征
        """
        features = {}

        try:
            # 度中心性
            features['degree_centrality'] = nx.degree_centrality(G).get(rid, 0)

            # 接近中心性
            if nx.is_weakly_connected(G):
                features['closeness_centrality'] = nx.closeness_centrality(G).get(rid, 0)
            else:
                features['closeness_centrality'] = 0

            # 介数中心性
            features['betweenness_centrality'] = nx.betweenness_centrality(G).get(rid, 0)

            # PageRank
            features['pagerank'] = nx.pagerank(G).get(rid, 0)

        except Exception as e:
            # 如果计算失败，返回默认值
            features['degree_centrality'] = 0
            features['closeness_centrality'] = 0
            features['betweenness_centrality'] = 0
            features['pagerank'] = 0

        return features

    def extract_node_structure_features(self, G, rid):
        """
        提取节点结构特征
        """
        features = {}

        # 入度和出度
        features['in_degree'] = G.in_degree(rid)
        features['out_degree'] = G.out_degree(rid)
        features['total_degree'] = features['in_degree'] + features['out_degree']

        # 邻居节点信息
        in_neighbors = list(G.predecessors(rid))
        out_neighbors = list(G.successors(rid))

        features['num_in_neighbors'] = len(in_neighbors)
        features['num_out_neighbors'] = len(out_neighbors)

        # 二跳邻居
        two_hop_neighbors = set()
        for neighbor in out_neighbors:
            two_hop_neighbors.update(G.successors(neighbor))
        features['num_two_hop_neighbors'] = len(two_hop_neighbors)

        # 聚类系数
        try:
            features['clustering_coefficient'] = nx.clustering(G.to_undirected(), rid)
        except:
            features['clustering_coefficient'] = 0

        return features

    def extract_node_attribute_features(self, node):
        """
        提取节点属性特征（类别、供应商等）
        """
        features = {}

        # 类别编码
        node_class = node.get('@class', 'Unknown')
        features['class_encoded'] = self.class_to_idx.get(node_class, -1)

        # 供应商编码
        vendor = node.get('vendor_name', 'Unknown')
        features['vendor_encoded'] = self.vendor_to_idx.get(vendor, -1)

        # 设备类型编码
        device_type = node.get('device_type', 'Unknown')
        features['device_encoded'] = self.device_to_idx.get(device_type, -1)

        return features

    def extract_graph_level_features(self, G):
        """
        提取图级别特征
        """
        features = {}

        # 图的基本统计
        features['num_nodes'] = G.number_of_nodes()
        features['num_edges'] = G.number_of_edges()
        features['density'] = nx.density(G)

        # 连通性
        features['num_weakly_connected'] = nx.number_weakly_connected_components(G)
        features['num_strongly_connected'] = nx.number_strongly_connected_components(G)

        # 平均度数
        degrees = [d for n, d in G.degree()]
        features['avg_degree'] = np.mean(degrees) if degrees else 0
        features['std_degree'] = np.std(degrees) if degrees else 0
        features['max_degree'] = np.max(degrees) if degrees else 0

        return features

    def extract_all_features(self, case_data):
        """
        提取单个案例的所有特征
        """
        nodes = case_data['nodes']
        edges = case_data['edges']

        # 构建NetworkX图
        G = self.build_networkx_graph(nodes, edges)

        # 图级别特征
        graph_features = self.extract_graph_level_features(G)

        # 节点级别特征
        node_features_list = []
        node_ids = []

        for rid, node in nodes.items():
            node_feats = {}

            # 结构特征
            struct_feats = self.extract_node_structure_features(G, rid)
            node_feats.update(struct_feats)

            # 中心性特征
            centrality_feats = self.extract_node_centrality_features(G, rid)
            node_feats.update(centrality_feats)

            # 属性特征
            attr_feats = self.extract_node_attribute_features(node)
            node_feats.update(attr_feats)

            node_features_list.append(node_feats)
            node_ids.append(rid)

        return {
            'case_id': case_data['case_id'],
            'graph_features': graph_features,
            'node_features': node_features_list,
            'node_ids': node_ids,
            'labels': case_data.get('labels')
        }

    def process_dataset(self, data_file, output_file):
        """
        处理整个数据集
        """
        # 加载数据
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        print(f"处理 {len(data)} 个案例...")

        # 提取特征
        processed_data = []
        for case_data in tqdm(data):
            features = self.extract_all_features(case_data)
            processed_data.append(features)

        # 保存特征
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)

        print(f"特征保存到: {output_file}")
        return processed_data

    def run(self):
        """
        执行完整的特征提取流程
        """
        print("=" * 50)
        print("开始特征提取")
        print("=" * 50)

        # 处理训练集
        print("\n处理训练集...")
        train_data_file = os.path.join(self.data_folder, 'train_data.pkl')
        train_output_file = os.path.join(self.output_dir, 'train_features.pkl')
        train_features = self.process_dataset(train_data_file, train_output_file)

        # 处理测试集
        print("\n处理测试集...")
        test_data_file = os.path.join(self.data_folder, 'test_data.pkl')
        test_output_file = os.path.join(self.output_dir, 'test_features.pkl')
        test_features = self.process_dataset(test_data_file, test_output_file)

        print("\n特征提取完成！")
        print("=" * 50)


if __name__ == '__main__':
    extractor = FeatureExtractor(data_dir='./')
    extractor.run()
