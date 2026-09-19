#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果格式化模块
功能：将预测结果转换为提交格式
输入：results文件夹中的预测结果
输出：符合提交要求的结果文件，保存到submit文件夹
"""

import json
import os
import pickle
from tqdm import tqdm
from collections import defaultdict


class ResultFormatter:
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.results_dir = os.path.join(data_dir, 'results')
        self.submit_dir = os.path.join(data_dir, 'submit')
        self.test_dir = os.path.join(data_dir, 'test')

        os.makedirs(self.submit_dir, exist_ok=True)

    def load_predictions(self):
        """
        加载预测结果
        """
        pred_file = os.path.join(self.results_dir, 'predictions.pkl')
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
        return predictions

    def load_test_data(self, case_id):
        """
        加载测试集原始数据，获取节点详细信息
        """
        topo_file = os.path.join(self.test_dir, case_id, f"{case_id}.log.topo.json")

        if not os.path.exists(topo_file):
            return None

        with open(topo_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def format_prediction_for_case(self, case_id, predictions, test_data):
        """
        为单个案例格式化预测结果
        生成符合比赛要求的rootcause格式
        """
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

            # 构建根因条目
            rootcause_entry = {
                "@rid": node_id,
                "title": self.generate_fault_title(node),
                "location": self.extract_location(node),
                "reason": self.generate_fault_reason(node),
                "confidence": pred['probability']  # 添加置信度（可选）
            }

            rootcause_list.append(rootcause_entry)

        return {
            "rootcause": rootcause_list
        }

    def generate_fault_title(self, node):
        """
        根据节点信息生成故障标题
        """
        node_class = node.get('@class', '未知设备')

        # 根据节点类别生成故障标题
        fault_titles = {
            'BaseStation': '基站故障',
            'TransCircuit': '传输电路故障',
            'Equipment': '设备故障',
            'PowerSupply': '供电故障',
            'Board': '单板故障',
        }

        # 默认故障类型
        title = fault_titles.get(node_class, '设备故障')

        # 如果有供应商信息，考虑供电相关故障
        if 'Power' in node_class or node.get('device_type') == '电源设备':
            title = '设备掉电'

        return title

    def extract_location(self, node):
        """
        提取位置信息
        """
        # 尝试从节点标签中提取位置
        label = node.get('zh_label', '')

        # 如果有其他位置字段，可以从这里提取
        # 这里使用简化的位置信息
        location = f"node_id={node.get('@rid', '')}"

        return location

    def generate_fault_reason(self, node):
        """
        根据节点信息生成故障原因
        """
        node_class = node.get('@class', '')

        # 根据节点类别生成可能的故障原因
        reasons = {
            'BaseStation': '1. 基站设备故障。\n2. 传输链路中断。\n3. 电源供电异常。',
            'TransCircuit': '1. 传输线路故障。\n2. 传输设备故障。\n3. 光纤链路中断。',
            'Equipment': '1. 设备硬件故障。\n2. 软件异常。\n3. 配置错误。',
            'PowerSupply': '1. 市电异常。\n2. 外部供电设备故障。\n3. 电源模块损坏。',
        }

        default_reason = '1. 市电异常。\n2. 外部供电设备故障。'
        reason = reasons.get(node_class, default_reason)

        return reason

    def save_formatted_results(self, case_id, formatted_result):
        """
        保存格式化后的结果
        """
        case_dir = os.path.join(self.submit_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)

        output_file = os.path.join(case_dir, f"{case_id}.rootcause.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_result, f, ensure_ascii=False, indent=4)

    def run(self):
        """
        执行完整的结果格式化流程
        """
        print("=" * 50)
        print("开始结果格式化")
        print("=" * 50)

        # 加载预测结果
        print("\n加载预测结果...")
        predictions = self.load_predictions()

        print(f"待处理的案例数: {len(predictions)}")

        # 格式化每个案例的结果
        success_count = 0
        for case_id, preds in tqdm(predictions.items()):
            # 加载测试数据
            test_data = self.load_test_data(case_id)

            if test_data is None:
                print(f"警告: 无法加载案例 {case_id} 的测试数据")
                continue

            # 格式化结果
            formatted_result = self.format_prediction_for_case(case_id, preds, test_data)

            if formatted_result:
                # 保存结果
                self.save_formatted_results(case_id, formatted_result)
                success_count += 1

        print(f"\n成功格式化 {success_count}/{len(predictions)} 个案例")
        print(f"结果已保存到: {self.submit_dir}")
        print("\n结果格式化完成！")
        print("=" * 50)


if __name__ == '__main__':
    formatter = ResultFormatter(data_dir='./')
    formatter.run()
