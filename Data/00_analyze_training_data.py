#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练数据分析模块
功能：深入分析训练集中的根因节点特征，找出高频故障模式
"""

import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm


class TrainingDataAnalyzer:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')

    def analyze(self):
        """执行分析"""
        print("=" * 60)
        print("训练数据深度分析")
        print("=" * 60)

        # 统计数据
        rootcause_titles = []
        rootcause_locations = []
        rootcause_reasons = []
        rootcause_node_classes = []
        rootcause_node_features = []

        case_dirs = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        print(f"\n总案例数: {len(case_dirs)}")

        for case_id in tqdm(case_dirs, desc="分析训练案例"):
            case_path = os.path.join(self.train_dir, case_id)
            topo_file = os.path.join(case_path, f"{case_id}.log.topo.json")
            rootcause_file = os.path.join(case_path, f"{case_id}.rootcause.json")

            if not os.path.exists(topo_file) or not os.path.exists(rootcause_file):
                continue

            # 加载数据
            with open(topo_file, 'r', encoding='utf-8') as f:
                topo_data = json.load(f)

            with open(rootcause_file, 'r', encoding='utf-8') as f:
                rootcause_data = json.load(f)

            # 构建节点映射
            nodes_map = {}
            for node in topo_data.get('nodes', []):
                rid = node.get('@rid')
                if rid:
                    nodes_map[rid] = node

            # 分析根因节点
            for rc in rootcause_data.get('rootcause', []):
                rid = rc.get('@rid')
                title = rc.get('title', '')
                location = rc.get('location', '')
                reason = rc.get('reason', '')

                rootcause_titles.append(title)
                rootcause_locations.append(location)
                rootcause_reasons.append(reason)

                # 如果根因节点在拓扑中
                if rid in nodes_map:
                    node = nodes_map[rid]
                    node_class = node.get('@class', 'Unknown')
                    rootcause_node_classes.append(node_class)

                    # 计算节点特征
                    in_degree = len(node.get('in_dependOn', []))
                    out_degree = len(node.get('out_dependOn', []))
                    rootcause_node_features.append({
                        'class': node_class,
                        'in_degree': in_degree,
                        'out_degree': out_degree,
                        'total_degree': in_degree + out_degree
                    })

        # 输出分析结果
        print("\n" + "=" * 60)
        print("根因标题分布 (Top 20):")
        print("=" * 60)
        title_counter = Counter(rootcause_titles)
        for title, count in title_counter.most_common(20):
            print(f"{title}: {count}")

        print("\n" + "=" * 60)
        print("根因节点类别分布:")
        print("=" * 60)
        class_counter = Counter(rootcause_node_classes)
        for node_class, count in class_counter.most_common():
            print(f"{node_class}: {count}")

        print("\n" + "=" * 60)
        print("根因节点度数统计:")
        print("=" * 60)
        if rootcause_node_features:
            avg_in = sum(f['in_degree'] for f in rootcause_node_features) / len(rootcause_node_features)
            avg_out = sum(f['out_degree'] for f in rootcause_node_features) / len(rootcause_node_features)
            avg_total = sum(f['total_degree'] for f in rootcause_node_features) / len(rootcause_node_features)

            print(f"平均入度: {avg_in:.2f}")
            print(f"平均出度: {avg_out:.2f}")
            print(f"平均总度数: {avg_total:.2f}")

        print("\n" + "=" * 60)
        print("位置信息关键词分析 (Top 20):")
        print("=" * 60)
        location_keywords = []
        for loc in rootcause_locations:
            if 'Equipment' in loc:
                location_keywords.append('Equipment')
            if 'Rack' in loc or 'rack' in loc:
                location_keywords.append('Rack')
            if 'ReplaceableUnit' in loc:
                location_keywords.append('ReplaceableUnit')
            if 'PlugInUnit' in loc or 'Slot' in loc:
                location_keywords.append('Board/Slot')
            if '框号' in loc or '槽号' in loc or '柜号' in loc:
                location_keywords.append('Hardware_Position')

        keyword_counter = Counter(location_keywords)
        for keyword, count in keyword_counter.most_common(20):
            print(f"{keyword}: {count}")

        print("\n" + "=" * 60)
        print("故障原因关键词分析 (Top 20):")
        print("=" * 60)
        reason_keywords = []
        for reason in rootcause_reasons:
            if '掉电' in reason or '电源' in reason:
                reason_keywords.append('Power_Issue')
            if 'CPRI' in reason or 'cpri' in reason:
                reason_keywords.append('CPRI_Issue')
            if '光' in reason or 'LOS' in reason:
                reason_keywords.append('Optical_Issue')
            if '单板' in reason or 'Board' in reason:
                reason_keywords.append('Board_Issue')
            if '链路' in reason or 'Link' in reason:
                reason_keywords.append('Link_Issue')

        reason_keyword_counter = Counter(reason_keywords)
        for keyword, count in reason_keyword_counter.most_common(20):
            print(f"{keyword}: {count}")

        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)


if __name__ == '__main__':
    analyzer = TrainingDataAnalyzer(data_dir='./')
    analyzer.analyze()
