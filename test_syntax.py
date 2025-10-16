#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本 - 检查项目中的语法错误
"""

import sys
import os

def test_imports():
    """测试导入是否正常"""
    try:
        print("测试导入numpy...")
        import numpy as np
        print("✓ numpy导入成功")
        
        print("测试导入matplotlib...")
        import matplotlib.pyplot as plt
        print("✓ matplotlib导入成功")
        
        print("测试导入tensorflow...")
        from tensorflow.keras.datasets import mnist
        print("✓ tensorflow导入成功")
        
        print("测试导入sklearn...")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        print("✓ sklearn导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_syntax():
    """测试语法是否正确"""
    try:
        print("测试main.py语法...")
        with open('main.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 尝试编译代码
        compile(code, 'main.py', 'exec')
        print("✓ main.py语法正确")
        
        print("测试extensions.py语法...")
        with open('extensions.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, 'extensions.py', 'exec')
        print("✓ extensions.py语法正确")
        
        print("测试unified_app.py语法...")
        with open('unified_app.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, 'unified_app.py', 'exec')
        print("✓ unified_app.py语法正确")
        
        return True
    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    try:
        print("测试神经网络类创建...")
        
        # 导入main模块
        import main
        
        # 创建神经网络实例
        model = main.NeuralNetwork(784, [128, 64], 10, 0.01)
        print("✓ 神经网络类创建成功")
        
        # 测试前向传播
        import numpy as np
        X_test = np.random.randn(10, 784)
        y_pred = model.forward_propagation(X_test)
        print(f"✓ 前向传播成功，输出形状: {y_pred.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试项目...")
    print("=" * 50)
    
    # 切换到项目目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 测试导入
    print("1. 测试依赖包导入...")
    import_success = test_imports()
    print()
    
    # 测试语法
    print("2. 测试代码语法...")
    syntax_success = test_syntax()
    print()
    
    # 测试基本功能
    if import_success and syntax_success:
        print("3. 测试基本功能...")
        func_success = test_basic_functionality()
        print()
        
        if func_success:
            print("🎉 所有测试通过！项目可以正常运行。")
        else:
            print("❌ 功能测试失败，可能存在运行时错误。")
    else:
        print("❌ 基础测试失败，请先解决导入或语法问题。")
    
    print("=" * 50)

