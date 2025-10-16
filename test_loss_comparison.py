#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的损失函数比较脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_comparison_analysis import ExperimentManager

def main():
    print("=" * 50)
    print("损失函数比较实验")
    print("=" * 50)
    print("比较以下两种损失函数：")
    print("1. 交叉熵损失 (Cross Entropy)")
    print("2. 均方误差 (MSE)")
    print("=" * 50)
    
    try:
        # 创建实验管理器
        print("正在初始化实验管理器...")
        manager = ExperimentManager()
        
        # 运行损失函数比较
        print("开始运行损失函数比较实验...")
        manager.compare_loss_functions()
        
        print("\n" + "=" * 50)
        print("实验完成！")
        print("结果文件：")
        print("- 损失函数_comparison.png (训练过程对比)")
        print("- 损失函数_test_performance.png (最终性能对比)")
        print("=" * 50)
        
    except Exception as e:
        print(f"运行实验时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
