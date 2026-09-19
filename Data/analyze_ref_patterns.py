#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析3个参考CSV文件（F1: 0.82, 0.81, 0.80）和我的生成版本
提取关键统计特征来指导算法优化
"""
import json
import csv
from collections import Counter, defaultdict

def analyze_csv(filepath, label):
    """分析单个CSV文件"""
    print(f"\n{'='*80}")
    print(f"分析: {label}")
    print('='*80)

    rc_counts = []
    titles = Counter()
    same_title_counts = []  # 记录相同title的根因数量

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            output = json.loads(row['output'])
            rcs = output.get('rootcause', [])
            rc_counts.append(len(rcs))

            # 统计title
            case_titles = Counter()
            for rc in rcs:
                title = rc.get('title', '')
                if title:
                    titles[title] += 1
                    case_titles[title] += 1

            # 记录这个案例中相同title出现的最大次数
            if case_titles:
                same_title_counts.append(max(case_titles.values()))

    # 基本统计
    total_cases = len(rc_counts)
    avg_rc = sum(rc_counts) / total_cases if total_cases > 0 else 0
    num_types = len(titles)
    rc_dist = Counter(rc_counts)

    print(f"总案例数: {total_cases}")
    print(f"平均根因数: {avg_rc:.2f}")
    print(f"故障类型数: {num_types}")

    print(f"\n根因数量分布:")
    for count in sorted(rc_dist.keys()):
        pct = rc_dist[count] / total_cases * 100
        print(f"  {count}个根因: {rc_dist[count]:3d}个案例 ({pct:5.1f}%)")

    print(f"\nTop 15 故障类型:")
    for title, cnt in titles.most_common(15):
        pct = cnt / sum(titles.values()) * 100
        print(f"  {title[:50]:50s}: {cnt:3d}次 ({pct:5.1f}%)")

    # 相同title统计
    avg_same = sum(same_title_counts) / len(same_title_counts) if same_title_counts else 0
    same_dist = Counter(same_title_counts)
    print(f"\n案例中相同title根因数量分布:")
    print(f"  平均每个案例最多有 {avg_same:.2f} 个相同title的根因")
    for count in sorted(same_dist.keys()):
        pct = same_dist[count] / len(same_title_counts) * 100
        print(f"  最多{count}个相同title: {same_dist[count]:3d}个案例 ({pct:5.1f}%)")

    return {
        'label': label,
        'total_cases': total_cases,
        'avg_rc': avg_rc,
        'num_types': num_types,
        'titles': titles,
        'rc_dist': rc_dist,
        'same_title_counts': same_title_counts
    }

# 分析3个参考文件
files = [
    ("/sessions/trusting-festive-allen/mnt/uploads/K-ON！20260729result_record (1).csv", "参考版本1 (F1=0.82)"),
    ("/sessions/trusting-festive-allen/mnt/uploads/K-ON！20260729result_record (2).csv", "参考版本2 (F1=0.81)"),
    ("/sessions/trusting-festive-allen/mnt/uploads/dbc65f85-dac4-4826-b8ce-8fbb67c78315-1785330152331_K-ON！20260729result_record.csv", "参考版本3 (F1=0.80)"),
]

all_stats = []
for filepath, label in files:
    stats = analyze_csv(filepath, label)
    all_stats.append(stats)

# 分析我的版本
print(f"\n{'='*80}")
print("分析我的生成版本")
print('='*80)

my_stats = analyze_csv("/sessions/trusting-festive-allen/mnt/Data/result_record.csv", "我的版本")

# 对比分析
print(f"\n{'='*80}")
print("对比分析")
print('='*80)

best = all_stats[0]  # F1=0.82的版本

print(f"\n1. 平均根因数对比:")
print(f"   最佳版本(F1=0.82): {best['avg_rc']:.2f}")
print(f"   我的版本:         {my_stats['avg_rc']:.2f}")
print(f"   差异:             {my_stats['avg_rc'] - best['avg_rc']:+.2f}")

print(f"\n2. 根因数量分布对比 (我的 vs 最佳):")
all_counts = sorted(set(list(best['rc_dist'].keys()) + list(my_stats['rc_dist'].keys())))
for count in all_counts:
    best_pct = best['rc_dist'].get(count, 0) / best['total_cases'] * 100
    my_pct = my_stats['rc_dist'].get(count, 0) / my_stats['total_cases'] * 100
    diff = my_pct - best_pct
    print(f"   {count}个根因: 最佳={best_pct:5.1f}% | 我的={my_pct:5.1f}% | 差异={diff:+6.1f}%")

print(f"\n3. 三个版本的平均根因数趋势:")
for s in all_stats:
    print(f"   {s['label']:25s}: {s['avg_rc']:.3f}")

print(f"\n{'='*80}")
print("优化建议")
print('='*80)

target_avg = best['avg_rc']
current_avg = my_stats['avg_rc']
diff = current_avg - target_avg

if abs(diff) <= 0.2:
    print(f"✓ 平均根因数接近目标（差异{diff:+.2f}），继续优化分布")
elif current_avg > target_avg:
    print(f"✗ 平均根因数偏高（差异{diff:+.2f}）")
    print(f"  建议:")
    print(f"  1. 降低K值计算公式的系数")
    print(f"  2. 提高筛选阈值（in-degree或其他指标）")
    print(f"  3. 检查是否选择了过多相同类型的根因")
else:
    print(f"✗ 平均根因数偏低（差异{diff:+.2f}）")
    print(f"  建议:")
    print(f"  1. 提高K值计算公式的系数")
    print(f"  2. 降低筛选阈值")

# 分析根因数为1的案例比例
print(f"\n4. 单根因案例比例:")
for s in all_stats:
    pct = s['rc_dist'].get(1, 0) / s['total_cases'] * 100
    print(f"   {s['label']:25s}: {pct:5.1f}%")
my_pct = my_stats['rc_dist'].get(1, 0) / my_stats['total_cases'] * 100
print(f"   {'我的版本':25s}: {my_pct:5.1f}%")

print(f"\n完成分析!")
