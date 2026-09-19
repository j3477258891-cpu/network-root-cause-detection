#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版特征提取模块
功能：从图结构中提取节点特征、图特征和语义特征
输入：data文件夹中的处理后数据
输出：增强的特征向量，保存到features文件夹
"""

import json
import os
import pickle
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import networkx as nx
import re


class EnhancedFeatureExtractor:
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
        """构建NetworkX图"""
        G = nx.DiGraph()
        for rid, node in nodes.items():
            G.add_node(rid, **node)
        for src, dst in edges:
            if src in G.nodes() and dst in G.nodes():
                G.add_edge(src, dst)
        return G

    def extract_semantic_features(self, node):
        """
        提取语义特征：从节点标签和属性中提取关键词特征
        """
        features = {}

        label = node.get('zh_label', '').lower()
        node_class = node.get('@class', '').lower()

        # 设备类型关键词
        features['is_base_station'] = 1 if 'basestation' in node_class else 0
        features['is_trans_circuit'] = 1 if 'trans' in node_class or 'circuit' in node_class else 0
        features['is_equipment'] = 1 if 'equipment' in node_class else 0
        features['is_board'] = 1 if 'board' in node_class or 'unit' in node_class else 0
        features['is_power'] = 1 if 'power' in node_class or 'power' in label else 0

        # 位置关键词
        features['has_rack'] = 1 if 'rack' in label or '机架' in label else 0
        features['has_shelf'] = 1 if 'shelf' in label or '框' in label else 0
        features['has_slot'] = 1 if 'slot' in label or '槽' in label else 0

        # 供应商特征
        vendor = node.get('vendor_name', '').lower()
        features['has_vendor'] = 1 if vendor and vendor != 'unknown' else 0

        return features

    def extract_topology_features(self, G, rid):
        """
        提取拓扑特征：节点在图中的位置和作用
        """
        features = {}

        # 基本度数特征
        features['in_degree'] = G.in_degree(rid)
        features['out_degree'] = G.out_degree(rid)
        features['total_degree'] = features['in_degree'] + features['out_degree']

        # 度数比率
        if features['total_degree'] > 0:
            features['in_out_ratio'] = features['in_degree'] / features['total_degree']
        else:
            features['in_out_ratio'] = 0

        # 邻居特征
        predecessors = list(G.predecessors(rid))
        successors = list(G.successors(rid))

        features['num_predecessors'] = len(predecessors)
        features['num_successors'] = len(successors)

        # 邻居的平均度数
        if predecessors:
            pred_degrees = [G.degree(p) for p in predecessors]
            features['avg_predecessor_degree'] = np.mean(pred_degrees)
        else:
            features['avg_predecessor_degree'] = 0

        if successors:
            succ_degrees = [G.degree(s) for s in successors]
            features['avg_successor_degree'] = np.mean(succ_degrees)
        else:
            features['avg_successor_degree'] = 0

        # K-core值（节点的核心性）
        try:
            k_core = nx.core_number(G.to_undirected())
            features['k_core'] = k_core.get(rid, 0)
        except:
            features['k_core'] = 0

        # 聚类系数
        try:
            features['clustering'] = nx.clustering(G.to_undirected(), rid)
        except:
            features['clustering'] = 0

        # 二跳邻居数量
        two_hop_neighbors = set()
        for succ in successors:
            two_hop_neighbors.update(G.successors(succ))
        features['two_hop_neighbors'] = len(two_hop_neighbors)

        return features

    def extract_centrality_features(self, G, rid):
        """提取中心性特征"""
        features = {}

        try:
            # 度中心性
            degree_centrality = nx.degree_centrality(G)
            features['degree_centrality'] = degree_centrality.get(rid, 0)

            # Betweenness中心性（采样计算，提高速度）
            if G.number_of_nodes() > 1000:
                # 大图采样计算
                k = min(100, G.number_of_nodes())
                betweenness = nx.betweenness_centrality(G, k=k)
            else:
                betweenness = nx.betweenness_centrality(G)
            features['betweenness_centrality'] = betweenness.get(rid, 0)

            # PageRank
            pagerank = nx.pagerank(G, max_iter=50)
            features['pagerank'] = pagerank.get(rid, 0)

            # Closeness中心性（仅对弱连通分量计算）
            if nx.is_weakly_connected(G):
                closeness = nx.closeness_centrality(G)
                features['closeness_centrality'] = closeness.get(rid, 0)
            else:
                # 计算节点所在连通分量的closeness
                try:
                    subgraph = G.subgraph(nx.node_connected_component(G.to_undirected(), rid))
                    closeness = nx.closeness_centrality(subgraph)
                    features['closeness_centrality'] = closeness.get(rid, 0)
                except:
                    features['closeness_centrality'] = 0

        except Exception as e:
            # 如果计算失败，使用默认值
            features['degree_centrality'] = 0
            features['betweenness_centrality'] = 0
            features['pagerank'] = 0
            features['closeness_centrality'] = 0

        return features

    def extract_attribute_features(self, node):
        """提取属性特征"""
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

    def extract_graph_features(self, G):
        """提取图级别特征"""
        features = {}

        features['num_nodes'] = G.number_of_nodes()
        features['num_edges'] = G.number_of_edges()
        features['density'] = nx.density(G)

        # 连通性
        features['num_weakly_connected'] = nx.number_weakly_connected_components(G)
        features['num_strongly_connected'] = nx.number_strongly_connected_components(G)

        # 度数统计
        degrees = [d for n, d in G.degree()]
        if degrees:
            features['avg_degree'] = np.mean(degrees)
            features['std_degree'] = np.std(degrees)
            features['max_degree'] = np.max(degrees)
            features['min_degree'] = np.min(degrees)
        else:
            features['avg_degree'] = 0
            features['std_degree'] = 0
            features['max_degree'] = 0
            features['min_degree'] = 0

        # 入度和出度统计
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]

        if in_degrees:
            features['avg_in_degree'] = np.mean(in_degrees)
            features['max_in_degree'] = np.max(in_degrees)
        else:
            features['avg_in_degree'] = 0
            features['max_in_degree'] = 0

        if out_degrees:
            features['avg_out_degree'] = np.mean(out_degrees)
            features['max_out_degree'] = np.max(out_degrees)
        else:
            features['avg_out_degree'] = 0
            features['max_out_degree'] = 0

        return features

    def extract_all_features(self, case_data):
        """提取单个案例的所有特征"""
        nodes = case_data['nodes']
        edges = case_data['edges']

        # 构建图
        G = self.build_networkx_graph(nodes, edges)

        # 图级别特征
        graph_features = self.extract_graph_features(G)

        # 节点级别特征
        node_features_list = []
        node_ids = []

        for rid, node in nodes.items():
            node_feats = {}

            # 拓扑特征
            topo_feats = self.extract_topology_features(G, rid)
            node_feats.update(topo_feats)

            # 中心性特征
            centrality_feats = self.extract_centrality_features(G, rid)
            node_feats.update(centrality_feats)

            # 属性特征
            attr_feats = self.extract_attribute_features(node)
            node_feats.update(attr_feats)

            # 语义特征
            semantic_feats = self.extract_semantic_features(node)
            node_feats.update(semantic_feats)

            node_features_list.append(node_feats)
            node_ids.append(rid)

        return {
            'case_id': case_data['case_id'],
            'graph_features': graph_features,
            'node_features': node_features_list,
            'node_ids': node_ids,
            'labels': case_data.get('labels'),
            'nodes': nodes  # 保留原始节点信息用于后处理
        }

    def process_dataset(self, data_file, output_file):
        """处理整个数据集"""
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        print(f"处理 {len(data)} 个案例...")

        processed_data = []
        for case_data in tqdm(data):
            features = self.extract_all_features(case_data)
            processed_data.append(features)

        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)

        print(f"特征保存到: {output_file}")
        return processed_data

    def run(self):
        """执行完整的特征提取流程"""
        print("=" * 50)
        print("开始增强特征提取")
        print("=" * 50)

        # 处理训练集
        print("\n处理训练集...")
        train_data_file = os.path.join(self.data_folder, 'train_data.pkl')
        train_output_file = os.path.join(self.output_dir, 'train_features_enhanced.pkl')
        self.process_dataset(train_data_file, train_output_file)

        # 处理测试集
        print("\n处理测试集...")
        test_data_file = os.path.join(self.data_folder, 'test_data.pkl')
        test_output_file = os.path.join(self.output_dir, 'test_features_enhanced.pkl')
        self.process_dataset(test_data_file, test_output_file)

        print("\n增强特征提取完成！")
        print("=" * 50)


if __name__ == '__main__':
    extractor = EnhancedFeatureExtractor(data_dir='./')
    extractor.run()
