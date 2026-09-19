#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主运行脚本
功能：一键运行完整的根因分析流程
"""

import os
import time
import subprocess
import sys


def run_script(script_name, description):
    """
    运行Python脚本并显示进度
    """
    print("\n" + "=" * 60)
    print(f"步骤: {description}")
    print("=" * 60)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )

        elapsed_time = time.time() - start_time
        print(f"\n✓ {description} 完成！耗时: {elapsed_time:.2f} 秒")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} 失败！")
        print(f"错误信息: {e}")
        return False


def main():
    """
    执行完整流程
    """
    print("=" * 60)
    print("网络故障根因分析系统")
    print("=" * 60)

    start_time = time.time()

    # 步骤1: 数据预处理
    if not run_script('01_data_preprocess.py', '数据预处理'):
        print("\n流程中断：数据预处理失败")
        return

    # 步骤2: 特征提取
    if not run_script('02_feature_extraction.py', '特征提取'):
        print("\n流程中断：特征提取失败")
        return

    # 步骤3: 模型训练
    if not run_script('03_model_training.py', '模型训练'):
        print("\n流程中断：模型训练失败")
        return

    # 步骤4: 模型预测
    if not run_script('04_model_prediction.py', '模型预测'):
        print("\n流程中断：模型预测失败")
        return

    # 步骤5: 结果格式化
    if not run_script('05_format_results.py', '结果格式化'):
        print("\n流程中断：结果格式化失败")
        return

    # 完成
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print("\n" + "=" * 60)
    print("✓ 所有步骤执行完成！")
    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")
    print("=" * 60)
    print("\n提交文件位置: ./submit/")
    print("每个测试案例的预测结果已保存为对应的 .rootcause.json 文件")


if __name__ == '__main__':
    main()
