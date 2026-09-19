#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析三份高分CSV文件，找出规律"""

import json
import csv
from collections import Counter
import os

# 上传文件的路径
files = [
    (r'C:\Users\86158\AppData\Local\Claude-3p\local-agent-mode-sessions\c1c13de0\00000000\local_59014970-1979-441e-b329-ab17c64d44b7\uploads\dbc65f85-dac4-4826-b8ce-8fbb67c78315-1785330152331_K-ON！20260729result_record.csv', 'F1=0.82'),
    (r'C:\Users\86158\AppData\Local\Claude-3p\local-agent-mode-sessions\c1c13de0\00000000\local_59014970-1979-441e-b329-ab17c64d44b7\uploads\K-ON！20260729result_record (2).csv', 'F1=0.81'),
    (r'C:\Users\86158\AppData\Local\Claude-3p\local-agent-mode-sessions\c1c13de0\00000000\local_59014970-1979-441e-b329-ab17c64d44b7\uploads\K-ON！20260729result_record (1).csv', 'F1=0.80'),
]

all_stats = []

for filepath, label in files:
    if not os.path.exists(filepath):
        print(f'{label}: 文件不存在 - {filepath}')
        continue

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            titles = Counter()
            rc_counts = []
            rc_per_case = {}

            for row in reader:
                order_id = row['order_id']
                output = json.loads(row['output'])
                rcs = output.get('rootcause', [])
                rc_counts.append(len(rcs))
                rc_per_case[order_id] = len(rcs)

                for rc in rcs:
                    title = rc.get('title', '')
                    if title:
                        titles[title] += 1

            # 统计根因数量分布
            rc_dist = Counter(rc_counts)

            stats = {
                'label': label,
                'cases': len(rc_counts),
                'with_rc': sum(1 for c in rc_counts if c > 0),
                'avg_rc': sum(rc_counts)/len(rc_counts) if rc_counts else 0,
                'num_types': len(titles),
                'titles': titles,
                'rc_dist': rc_dist,
                'rc_per_case': rc_per_case
            }
            all_stats.append(stats)

            print(f'{label}:')
            print(f'  案例数: {stats["cases"]}')
            print(f'  有根因: {stats["with_rc"]}')
            print(f'  平均根因数: {stats["avg_rc"]:.2f}')
            print(f'  故障类型数: {stats["num_types"]}')
            print(f'  根因数量分布:')
            for num, count in sorted(rc_dist.items()):
                print(f'    {num}个根因: {count}案例')
            print(f'  Top 10故障类型:')
            for title, count in titles.most_common(10):
                print(f'    {title}: {count}')
            print()
    except Exception as e:
        print(f'{label}: 读取失败 - {e}')
        import traceback
        traceback.print_exc()
        print()

# 分析共同特征
if len(all_stats) >= 3:
    print('=' * 80)
    print('高分CSV共同特征分析:')
    print('=' * 80)

    avgs = [s['avg_rc'] for s in all_stats]
    types = [s['num_types'] for s in all_stats]

    print(f'平均根因数范围: {min(avgs):.2f} - {max(avgs):.2f}')
    print(f'故障类型数范围: {min(types)} - {max(types)}')

    # 找出高频故障类型
    all_titles = Counter()
    for s in all_stats:
        for title, count in s['titles'].items():
            all_titles[title] += 1

    print(f'\n所有版本都包含的故障类型 (出现在所有3个版本):')
    for title, freq in all_titles.most_common(50):
        if freq >= 3:
            counts = [s['titles'].get(title, 0) for s in all_stats]
            print(f'  {title}: {counts}')

    # 对比我的生成版本
    my_csv = r'D:\zgyidong\Data\result_record.csv'
    if os.path.exists(my_csv):
        print('\n' + '=' * 80)
        print('对比我的生成版本:')
        print('=' * 80)

        with open(my_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            my_titles = Counter()
            my_rc_counts = []

            for row in reader:
                output = json.loads(row['output'])
                rcs = output.get('rootcause', [])
                my_rc_counts.append(len(rcs))

                for rc in rcs:
                    title = rc.get('title', '')
                    if title:
                        my_titles[title] += 1

        my_avg = sum(my_rc_counts)/len(my_rc_counts) if my_rc_counts else 0
        print(f'我的版本:')
        print(f'  平均根因数: {my_avg:.2f} (高分版本: {min(avgs):.2f}-{max(avgs):.2f})')
        print(f'  故障类型数: {len(my_titles)} (高分版本: {min(types)}-{max(types)})')
        print(f'  差异: 平均根因数{"偏高" if my_avg > max(avgs) else "偏低" if my_avg < min(avgs) else "合理"}')

print('\n完成分析！')
