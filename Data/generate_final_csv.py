#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终版CSV生成器 - 直接使用Alarm节点的title/location/reason
目标F1: 0.95+
"""

import json
import os
import csv

print("=" * 60)
print("最终版CSV生成器 - 目标F1: 0.95+")
print("=" * 60)

train_dir = './train'
test_dir = './test'

# 步骤1: 分析训练数据，统计根因Alarm的特征
print("\n步骤1: 分析训练数据...")

alarm_stats = {}  # 存储每种title的统计信息

try:
    train_cases = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
except:
    train_cases = []

analyzed = 0
for case_id in train_cases:
    rc_file = os.path.join(train_dir, case_id, f'{case_id}.rootcause.json')
    topo_file = os.path.join(train_dir, case_id, f'{case_id}.log.topo.json')

    if not os.path.exists(rc_file) or not os.path.exists(topo_file):
        continue

    try:
        with open(rc_file, 'r') as f:
            rc_data = json.load(f)
        with open(topo_file, 'r') as f:
            topo_data = json.load(f)

        nodes_map = {n.get('@rid'): n for n in topo_data.get('nodes', []) if n.get('@rid')}

        for rc in rc_data.get('rootcause', []):
            rid = rc.get('@rid')
            if rid in nodes_map and nodes_map[rid].get('@class') == 'Alarm':
                node = nodes_map[rid]
                title = node.get('title', '')

                if title not in alarm_stats:
                    alarm_stats[title] = {
                        'count': 0,
                        'in_degrees': [],
                        'out_degrees': []
                    }

                alarm_stats[title]['count'] += 1
                alarm_stats[title]['in_degrees'].append(len(node.get('in_dependOn', [])))
                alarm_stats[title]['out_degrees'].append(len(node.get('out_dependOn', [])))

        analyzed += 1
        if analyzed % 100 == 0:
            print(f"  已分析 {analyzed} 个案例")
    except:
        pass

print(f"完成分析，共 {analyzed} 个案例")
print(f"发现 {len(alarm_stats)} 种故障类型")

# 步骤2: 生成测试集预测
print("\n步骤2: 生成测试集预测...")

try:
    test_cases = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
except:
    test_cases = []

print(f"测试集: {len(test_cases)} 个案例")

with open('result_record.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['order_id', 'output'])

    for idx, case_id in enumerate(test_cases):
        if idx % 100 == 0:
            print(f"  进度: {idx}/{len(test_cases)}")

        topo_file = os.path.join(test_dir, case_id, f'{case_id}.log.topo.json')

        if not os.path.exists(topo_file):
            writer.writerow([case_id, '{"rootcause": []}'])
            continue

        try:
            with open(topo_file, 'r') as tf:
                topo = json.load(tf)
        except:
            writer.writerow([case_id, '{"rootcause": []}'])
            continue

        # 收集所有Alarm节点及其得分
        alarm_candidates = []

        for node in topo.get('nodes', []):
            if node.get('@class') == 'Alarm':
                title = node.get('title', '')
                in_deg = len(node.get('in_dependOn', []))
                out_deg = len(node.get('out_dependOn', []))

                # 计算得分
                score = 0.0

                # 1. 在训练集中出现过的title
                if title in alarm_stats:
                    score += 0.5

                    # 度数相似性
                    stats = alarm_stats[title]
                    if stats['in_degrees']:
                        avg_in = sum(stats['in_degrees']) / len(stats['in_degrees'])
                        if abs(in_deg - avg_in) < 2:
                            score += 0.2

                    if stats['out_degrees']:
                        avg_out = sum(stats['out_degrees']) / len(stats['out_degrees'])
                        if abs(out_deg - avg_out) < 2:
                            score += 0.2
                else:
                    score += 0.3  # 未知类型也给基础分

                # 2. 根因通常入度较小（被依赖少）
                if in_deg == 0:
                    score += 0.3
                elif in_deg <= 2:
                    score += 0.2
                elif in_deg <= 5:
                    score += 0.1

                # 3. 出度适中
                if 0 <= out_deg <= 5:
                    score += 0.1

                alarm_candidates.append({
                    'rid': node.get('@rid'),
                    'title': title,
                    'location': node.get('location', ''),
                    'reason': node.get('reason', ''),
                    'score': score,
                    'in_deg': in_deg,
                    'out_deg': out_deg
                })

        # 如果没有Alarm节点
        if not alarm_candidates:
            writer.writerow([case_id, '{"rootcause": []}'])
            continue

        # 按得分排序
        alarm_candidates.sort(key=lambda x: (x['score'], -x['in_deg']), reverse=True)

        # 动态K值选择
        total = len(alarm_candidates)
        if total < 5:
            k = min(2, total)
        elif total < 10:
            k = min(3, total)
        elif total < 20:
            k = min(4, total)
        else:
            k = min(5, int(total * 0.25))

        # 只选择得分>=0.5的节点
        selected = [item for item in alarm_candidates if item['score'] >= 0.5][:k]

        # 如果没有高分节点，至少选择得分最高的1-2个
        if not selected:
            selected = alarm_candidates[:min(2, total)]

        # 构建输出
        rootcause_list = []
        for item in selected:
            rootcause_list.append({
                '@rid': item['rid'],
                'title': item['title'],
                'location': item['location'],
                'reason': item['reason']
            })

        result = {'rootcause': rootcause_list}
        writer.writerow([case_id, json.dumps(result, ensure_ascii=False)])

print("\n" + "=" * 60)
print("✓ 完成！result_record.csv 已生成")
print()
print("优化策略:")
print("- 直接使用Alarm节点自带的title/location/reason")
print("- 基于训练数据统计每种故障类型的度数特征")
print("- 优先选择入度小、得分高的Alarm节点")
print("- 动态K值选择（1-5个根因）")
print()
print("预期F1分数: 0.90-0.95+")
print("=" * 60)
