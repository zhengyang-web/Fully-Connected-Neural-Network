#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化测试脚本
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """测试基本导入"""
    try:
        print("测试基本导入...")
        import numpy as np
        print("✓ numpy导入成功")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_neural_network_creation():
    """测试神经网络创建"""
    try:
        print("测试神经网络类...")
        
        # 导入main模块
        import main
        
        # 创建简单的神经网络
        model = main.NeuralNetwork(784, [128, 64], 10, 0.01)
        print("✓ 神经网络创建成功")
        
        return True
    except Exception as e:
        print(f"✗ 神经网络创建失败: {e}")
        return False

def test_forward_propagation():
    """测试前向传播"""
    try:
        print("测试前向传播...")
        
        import main
        import numpy as np
        
        # 创建模型
        model = main.NeuralNetwork(784, [128, 64], 10, 0.01)
        
        # 创建测试数据
        X_test = np.random.randn(5, 784)
        
        # 前向传播
        y_pred = model.forward_propagation(X_test)
        print(f"✓ 前向传播成功，输出形状: {y_pred.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False

if __name__ == "__main__":
    print("开始简化测试...")
    print("=" * 40)
    
    # 测试导入
    if not test_basic_imports():
        print("❌ 基础导入失败")
        sys.exit(1)
    
    # 测试神经网络创建
    if not test_neural_network_creation():
        print("❌ 神经网络创建失败")
        sys.exit(1)
    
    # 测试前向传播
    if not test_forward_propagation():
        print("❌ 前向传播失败")
        sys.exit(1)
    
    print("=" * 40)
    print("🎉 所有测试通过！")

