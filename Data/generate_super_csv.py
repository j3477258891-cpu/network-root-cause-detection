#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超强版根因预测 - 目标F1: 0.95+
策略：深度分析训练数据，提取根因节点的精确特征模式
"""

import json
import os
import pickle
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm


class SuperRootCausePredictor:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')

        # 根因模式库
        self.rootcause_patterns = {}
        self.title_mapping = {}
        self.location_patterns = {}

    def analyze_training_patterns(self):
        """深度分析训练数据，提取根因节点的精确特征"""
        print("分析训练数据中的根因模式...")

        rootcause_features = []
        case_dirs = [d for d in os.listdir(self.train_dir)
                     if os.path.isdir(os.path.join(self.train_dir, d))]

        for case_id in tqdm(case_dirs[:500]):  # 分析前500个案例
            case_path = os.path.join(self.train_dir, case_id)
            topo_file = os.path.join(case_path, f"{case_id}.log.topo.json")
            rootcause_file = os.path.join(case_path, f"{case_id}.rootcause.json")

            if not os.path.exists(topo_file) or not os.path.exists(rootcause_file):
                continue

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

            # 分析根因节点特征
            for rc in rootcause_data.get('rootcause', []):
                rid = rc.get('@rid')
                if rid in nodes_map:
                    node = nodes_map[rid]

                    # 提取关键特征
                    pattern = {
                        'class': node.get('@class', ''),
                        'in_degree': len(node.get('in_dependOn', [])),
                        'out_degree': len(node.get('out_dependOn', [])),
                        'label_keywords': self.extract_keywords(node.get('zh_label', '')),
                        'title': rc.get('title', ''),
                        'location': rc.get('location', ''),
                        'reason': rc.get('reason', '')
                    }

                    rootcause_features.append(pattern)

        # 统计最常见的根因模式
        class_counter = Counter([p['class'] for p in rootcause_features])
        title_counter = Counter([p['title'] for p in rootcause_features])

        print(f"\n最常见的根因节点类别:")
        for cls, count in class_counter.most_common(10):
            print(f"  {cls}: {count}")

        print(f"\n最常见的故障标题:")
        for title, count in title_counter.most_common(10):
            print(f"  {title}: {count}")

        # 构建模式库
        for pattern in rootcause_features:
            cls = pattern['class']
            if cls not in self.rootcause_patterns:
                self.rootcause_patterns[cls] = []
            self.rootcause_patterns[cls].append(pattern)

        # 构建标题映射
        for pattern in rootcause_features:
            cls = pattern['class']
            title = pattern['title']
            if cls not in self.title_mapping:
                self.title_mapping[cls] = Counter()
            self.title_mapping[cls][title] += 1

        return rootcause_features

    def extract_keywords(self, text):
        """提取文本关键词"""
        keywords = []
        text_lower = text.lower()

        keyword_list = ['power', 'board', 'rack', 'shelf', 'slot', 'equipment',
                       'unit', 'trans', 'circuit', 'base', 'station',
                       '电源', '单板', '机架', '框', '槽', '设备']

        for kw in keyword_list:
            if kw in text_lower:
                keywords.append(kw)

        return keywords

    def calculate_rootcause_score(self, node, nodes_map):
        """
        计算节点作为根因的得分
        基于训练数据的模式匹配
        """
        score = 0.0
        node_class = node.get('@class', '')
        label = node.get('zh_label', '')
        in_degree = len(node.get('in_dependOn', []))
        out_degree = len(node.get('out_dependOn', []))

        # 1. 类别匹配得分
        if node_class in self.rootcause_patterns:
            score += 0.4

            # 度数相似性
            patterns = self.rootcause_patterns[node_class]
            avg_in = np.mean([p['in_degree'] for p in patterns])
            avg_out = np.mean([p['out_degree'] for p in patterns])

            # 度数越接近，得分越高
            in_diff = abs(in_degree - avg_in)
            out_diff = abs(out_degree - avg_out)

            if in_diff < 2:
                score += 0.1
            if out_diff < 2:
                score += 0.1

        # 2. 关键词匹配
        keywords = self.extract_keywords(label)
        if keywords:
            score += 0.2

        # 3. 特定类别加权
        high_prob_classes = ['ReplaceableUnit', 'PlugInUnit', 'SdrDeviceGroup',
                            'Equipment', 'BaseStation']
        if any(cls in node_class for cls in high_prob_classes):
            score += 0.3

        # 4. 度数特征
        # 根因节点通常入度较小（被依赖少），出度较大（依赖多个设备）
        if in_degree <= 2:
            score += 0.1
        if out_degree >= 1:
            score += 0.1

        # 5. 标签关键词
        label_lower = label.lower()
        if any(kw in label_lower for kw in ['power', 'board', 'rack', 'slot',
                                              '电源', '单板', '机架', '槽']):
            score += 0.2

        return min(score, 1.0)

    def predict_case(self, case_id):
        """预测单个案例"""
        topo_file = os.path.join(self.test_dir, case_id, f"{case_id}.log.topo.json")

        if not os.path.exists(topo_file):
            return []

        with open(topo_file, 'r', encoding='utf-8') as f:
            topo_data = json.load(f)

        # 构建节点映射
        nodes_map = {}
        for node in topo_data.get('nodes', []):
            rid = node.get('@rid')
            if rid:
                nodes_map[rid] = node

        # 计算每个节点的根因得分
        scores = {}
        for rid, node in nodes_map.items():
            score = self.calculate_rootcause_score(node, nodes_map)
            scores[rid] = score

        # 选择Top-K个节点
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 动态K值
        num_nodes = len(nodes_map)
        if num_nodes < 50:
            k = 1
        elif num_nodes < 100:
            k = 2
        elif num_nodes < 200:
            k = 3
        else:
            k = min(5, max(1, int(num_nodes * 0.02)))

        # 只选择得分>0.5的节点
        selected = [(rid, score) for rid, score in sorted_nodes[:k] if score > 0.5]

        # 如果没有高分节点，至少选择得分最高的1个
        if not selected and sorted_nodes:
            selected = [sorted_nodes[0]]

        predictions = []
        for rid, score in selected:
            node = nodes_map[rid]
            predictions.append({
                'rid': rid,
                'score': score,
                'node': node,
                'title': self.get_fault_title(node),
                'location': self.get_location(node),
                'reason': self.get_fault_reason(node)
            })

        return predictions

    def get_fault_title(self, node):
        """根据节点类别获取故障标题"""
        node_class = node.get('@class', '')

        # 使用训练数据中最常见的标题
        if node_class in self.title_mapping:
            most_common_title = self.title_mapping[node_class].most_common(1)[0][0]
            return most_common_title

        # 默认映射
        title_map = {
            'ReplaceableUnit': '输入电源断',
            'PlugInUnit': '单板不在位',
            'SdrDeviceGroup': '设备掉电',
            'Equipment': '设备掉电',
            'BaseStation': '设备掉电',
            'TransCircuit': '[衍生告警]PTN光缆中断，单报LOS'
        }

        for key, title in title_map.items():
            if key in node_class:
                return title

        return '设备故障'

    def get_location(self, node):
        """提取位置信息"""
        label = node.get('zh_label', '')
        if label:
            return label
        return node.get('@rid', '')

    def get_fault_reason(self, node):
        """获取故障原因"""
        node_class = node.get('@class', '')

        reason_map = {
            '输入电源断': '外部掉电',
            '设备掉电': '1. 市电异常。\n2. 外部供电设备故障。',
            '单板不在位': '1. OMC有配置的槽位，但实际没有插单板。\n2. OMC有配置的槽位，但单板没有插紧。',
            '[衍生告警]PTN光缆中断，单报LOS': '以太网物理接口(ETPI) 信号丢失(LOS)'
        }

        title = self.get_fault_title(node)
        return reason_map.get(title, '设备故障')

    def generate_csv(self, output_file='result_record.csv'):
        """生成CSV提交文件"""
        print("\n生成CSV文件...")

        test_cases = [d for d in os.listdir(self.test_dir)
                     if os.path.isdir(os.path.join(self.test_dir, d))]

        print(f"测试集案例数: {len(test_cases)}")

        import csv
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['order_id', 'output'])

            for case_id in tqdm(sorted(test_cases)):
                predictions = self.predict_case(case_id)

                rootcause_list = []
                for pred in predictions:
                    rootcause_list.append({
                        '@rid': pred['rid'],
                        'title': pred['title'],
                        'location': pred['location'],
                        'reason': pred['reason']
                    })

                result = {'rootcause': rootcause_list}
                output_json = json.dumps(result, ensure_ascii=False)
                writer.writerow([case_id, output_json])

        print(f"\nCSV文件已保存: {output_file}")

    def run(self):
        """执行完整流程"""
        print("=" * 60)
        print("超强版根因分析 - 目标F1: 0.95+")
        print("=" * 60)

        # 分析训练数据
        self.analyze_training_patterns()

        # 生成预测CSV
        output_file = os.path.join(self.data_dir, 'result_record.csv')
        self.generate_csv(output_file)

        print("\n" + "=" * 60)
        print("完成！请提交 result_record.csv 文件")
        print("=" * 60)


if __name__ == '__main__':
    predictor = SuperRootCausePredictor(data_dir='./')
    predictor.run()
