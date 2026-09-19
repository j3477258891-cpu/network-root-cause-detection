#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速CSV生成器 - 基于训练数据分析
目标F1: 0.95+
"""

import json
import os
import csv
from collections import Counter

print("=" * 60)
print("快速生成result_record.csv")
print("=" * 60)

# 步骤1: 快速分析训练数据
print("\n步骤1: 分析训练数据...")
train_dir = './train'
test_dir = './test'

try:
    train_cases = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))][:300]
except:
    train_cases = []

class_title_map = {}
class_location_map = {}
class_reason_map = {}
class_stats = {}

analyzed = 0
for case_id in train_cases:
    rc_file = os.path.join(train_dir, case_id, f'{case_id}.rootcause.json')
    topo_file = os.path.join(train_dir, case_id, f'{case_id}.log.topo.json')

    if not os.path.exists(rc_file) or not os.path.exists(topo_file):
        continue

    try:
        with open(rc_file, 'r', encoding='utf-8') as f:
            rc_data = json.load(f)
        with open(topo_file, 'r', encoding='utf-8') as f:
            topo_data = json.load(f)

        nodes_map = {n.get('@rid'): n for n in topo_data.get('nodes', []) if n.get('@rid')}

        for rc in rc_data.get('rootcause', []):
            rid = rc.get('@rid')
            if rid in nodes_map:
                node = nodes_map[rid]
                node_class = node.get('@class', 'Unknown')

                # 统计标题
                if node_class not in class_title_map:
                    class_title_map[node_class] = Counter()
                class_title_map[node_class][rc.get('title', '')] += 1

                # 统计原因
                if node_class not in class_reason_map:
                    class_reason_map[node_class] = Counter()
                class_reason_map[node_class][rc.get('reason', '')] += 1

                # 统计度数
                if node_class not in class_stats:
                    class_stats[node_class] = {'in': [], 'out': [], 'count': 0}
                class_stats[node_class]['count'] += 1
                class_stats[node_class]['in'].append(len(node.get('in_dependOn', [])))
                class_stats[node_class]['out'].append(len(node.get('out_dependOn', [])))

        analyzed += 1
        if analyzed % 50 == 0:
            print(f"  已分析 {analyzed} 个训练案例")
    except Exception as e:
        pass

print(f"完成训练数据分析，共 {analyzed} 个案例")
print(f"发现 {len(class_title_map)} 种根因类别")

# 步骤2: 生成测试集预测
print("\n步骤2: 生成测试集预测...")

try:
    test_cases = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
except:
    test_cases = []

print(f"测试集案例数: {len(test_cases)}")

output_file = 'result_record.csv'

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['order_id', 'output'])

    for idx, case_id in enumerate(test_cases):
        if idx % 50 == 0:
            print(f"  预测进度: {idx}/{len(test_cases)}")

        topo_file = os.path.join(test_dir, case_id, f'{case_id}.log.topo.json')

        if not os.path.exists(topo_file):
            # 空预测
            result = {'rootcause': []}
            writer.writerow([case_id, json.dumps(result, ensure_ascii=False)])
            continue

        try:
            with open(topo_file, 'r', encoding='utf-8') as tf:
                topo_data = json.load(tf)
        except:
            result = {'rootcause': []}
            writer.writerow([case_id, json.dumps(result, ensure_ascii=False)])
            continue

        # 计算每个节点的得分
        nodes_map = {n.get('@rid'): n for n in topo_data.get('nodes', []) if n.get('@rid')}
        scores = []

        for rid, node in nodes_map.items():
            node_class = node.get('@class', 'Unknown')
            score = 0.0

            # 在训练集中出现过的类别
            if node_class in class_stats:
                score += 0.6

                # 度数匹配
                in_deg = len(node.get('in_dependOn', []))
                out_deg = len(node.get('out_dependOn', []))

                stats = class_stats[node_class]
                if stats['in']:
                    avg_in = sum(stats['in']) / len(stats['in'])
                    if abs(in_deg - avg_in) < 3:
                        score += 0.2
                if stats['out']:
                    avg_out = sum(stats['out']) / len(stats['out'])
                    if abs(out_deg - avg_out) < 3:
                        score += 0.2

            # Alarm类别特殊处理（最高频）
            if node_class == 'Alarm':
                score += 0.3

            scores.append((rid, score, node))

        # 排序并选择Top-K
        scores.sort(key=lambda x: x[1], reverse=True)

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

        # 选择得分>0.5的节点
        selected = [(rid, score, node) for rid, score, node in scores[:k*2] if score > 0.5]
        if not selected and scores:
            selected = scores[:min(2, len(scores))]

        selected = selected[:k]

        # 构建输出
        rootcause_list = []
        for rid, score, node in selected:
            node_class = node.get('@class', 'Unknown')

            # 获取最常见的标题
            if node_class in class_title_map:
                title = class_title_map[node_class].most_common(1)[0][0]
            else:
                title = '设备故障'

            # 获取位置
            location = node.get('zh_label', rid)

            # 获取原因
            if node_class in class_reason_map:
                reason = class_reason_map[node_class].most_common(1)[0][0]
            else:
                reason = '设备故障'

            rootcause_list.append({
                '@rid': rid,
                'title': title,
                'location': location,
                'reason': reason
            })

        result = {'rootcause': rootcause_list}
        writer.writerow([case_id, json.dumps(result, ensure_ascii=False)])

print(f"\n完成！CSV文件已保存: {output_file}")
print("=" * 60)
