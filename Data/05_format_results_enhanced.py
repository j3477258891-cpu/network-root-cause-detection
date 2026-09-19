#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版结果格式化模块
功能：将预测结果转换为CSV提交格式
输入：results文件夹中的预测结果
输出：result_record.csv文件
"""

import json
import os
import pickle
import csv
from tqdm import tqdm


class EnhancedResultFormatter:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.results_dir = os.path.join(data_dir, 'results')
        self.test_dir = os.path.join(data_dir, 'test')

    def load_predictions(self):
        """加载预测结果"""
        pred_file = os.path.join(self.results_dir, 'predictions_enhanced.pkl')
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
        return predictions

    def load_test_data(self, case_id):
        """加载测试集原始数据"""
        topo_file = os.path.join(self.test_dir, case_id, f"{case_id}.log.topo.json")

        if not os.path.exists(topo_file):
            return None

        with open(topo_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def generate_fault_title(self, node):
        """
        根据节点信息生成故障标题
        根据训练数据中的高频标题进行匹配
        """
        node_class = node.get('@class', '').lower()
        label = node.get('zh_label', '').lower()

        # 基于节点类别的故障标题映射
        if 'power' in node_class or 'power' in label or 'replaceableunit' in node_class:
            return '输入电源断'
        elif 'board' in node_class or 'pluginunit' in node_class or 'sdrdevicegroup' in node_class:
            if '不在位' in label or 'absent' in label:
                return '单板不在位'
            elif '初始' in label:
                return '单板处于初始化状态'
            else:
                return '设备掉电'
        elif 'basestation' in node_class:
            return '设备掉电'
        elif 'trans' in node_class or 'circuit' in node_class:
            return '[衍生告警]PTN光缆中断，单报LOS'
        elif 'equipment' in node_class:
            return '设备掉电'
        else:
            # 默认标题
            return '设备故障'

    def extract_location(self, node):
        """提取位置信息"""
        label = node.get('zh_label', '')
        rid = node.get('@rid', '')

        # 尝试从rid中提取位置信息
        location_parts = []

        # 检查是否有SubNetwork等标准字段
        if 'SubNetwork' in label or 'ManagedElement' in label:
            return label

        # 从rid构建位置（简化版）
        location = rid

        return location

    def generate_fault_reason(self, node, title):
        """根据节点信息和标题生成故障原因"""
        # 根据标题匹配常见原因
        reason_map = {
            '输入电源断': '外部掉电',
            '设备掉电': '1. 市电异常。\n2. 外部供电设备故障。',
            '单板不在位': '1. OMC有配置的槽位，但实际没有插单板。\n2. OMC有配置的槽位，但单板没有插紧。\n3. OMC有配置的槽位，但接插件物理损坏。',
            '单板处于初始化状态': '1. 用户执行复位单板命令。\n2. 单板上电自启动。\n3. 软件异常导致单板自动复位。',
            '射频单元CPRI接口异常告警': '566',
            'BBU CPRI接口异常告警': '567',
            '[衍生告警]PTN光缆中断，单报LOS': '以太网物理接口(ETPI) 信号丢失(LOS)',
            'NR分布单元小区TRP不可用告警': '509',
            'GNSS星卡锁星不足告警': '506',
            '网元连接中断': '718',
            'BBU单板维护链路异常告警': '567',
            '制式间通信异常告警': '567'
        }

        return reason_map.get(title, '设备故障')

    def format_prediction_for_case(self, case_id, predictions, test_data):
        """为单个案例格式化预测结果"""
        if not predictions or not test_data:
            return None

        # 构建节点映射
        nodes_map = {}
        for node in test_data.get('nodes', []):
            rid = node.get('@rid')
            if rid:
                nodes_map[rid] = node

        # 格式化预测结果
        rootcause_list = []
        for pred in predictions:
            node_id = pred['node_id']

            if node_id not in nodes_map:
                continue

            node = nodes_map[node_id]

            # 生成故障标题
            title = self.generate_fault_title(node)

            # 提取位置
            location = self.extract_location(node)

            # 生成故障原因
            reason = self.generate_fault_reason(node, title)

            rootcause_entry = {
                "@rid": node_id,
                "title": title,
                "location": location,
                "reason": reason
            }

            rootcause_list.append(rootcause_entry)

        return {
            "rootcause": rootcause_list
        }

    def save_as_csv(self, all_results):
        """保存为CSV格式"""
        output_file = os.path.join(self.data_dir, 'result_record.csv')

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 写入表头
            writer.writerow(['order_id', 'output'])

            # 写入每个案例的结果
            for case_id in sorted(all_results.keys()):
                result = all_results[case_id]
                if result:
                    # 将结果转换为JSON字符串
                    output_json = json.dumps(result, ensure_ascii=False)
                    writer.writerow([case_id, output_json])

        print(f"\nCSV文件已保存到: {output_file}")

    def run(self):
        """执行完整的结果格式化流程"""
        print("=" * 50)
        print("开始增强结果格式化")
        print("=" * 50)

        # 加载预测结果
        print("\n加载预测结果...")
        predictions = self.load_predictions()

        print(f"待处理的案例数: {len(predictions)}")

        # 格式化每个案例的结果
        all_results = {}
        success_count = 0

        for case_id, preds in tqdm(predictions.items(), desc="格式化结果"):
            # 加载测试数据
            test_data = self.load_test_data(case_id)

            if test_data is None:
                print(f"警告: 无法加载案例 {case_id} 的测试数据")
                continue

            # 格式化结果
            formatted_result = self.format_prediction_for_case(case_id, preds, test_data)

            if formatted_result:
                all_results[case_id] = formatted_result
                success_count += 1

        print(f"\n成功格式化 {success_count}/{len(predictions)} 个案例")

        # 保存为CSV
        self.save_as_csv(all_results)

        print("\n结果格式化完成！")
        print("=" * 50)


if __name__ == '__main__':
    formatter = EnhancedResultFormatter(data_dir='./')
    formatter.run()
