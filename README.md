# Fully-Connected-Neural-Network
A1: 全连接网络 (20 points)  • 任务：训练一个全连接神经网络分类器，完成图像分类(20 points)   • 数据：MNIST手写数字体识别 或 CIFAR-10  • 要求：前馈、反馈、评估等代码需自己动手用numpy实现（而非使用pytorch）    附加题：   – 尝试使用不同的损失函数和正则化方法，观察并分析其对实验结果的影响 (+5 points)  – 尝试使用不同的优化算法，观察并分析其对训练过程和实验结果的影响 (如batch GD, online GD, mini-batch GD, SGD, 或其它的优化算法如Momentum, Adsgrad, Adam, Admax) (+5 points)
# 全连接神经网络实现 - MNIST手写数字识别

## 项目简介
这是我在机器学习课程中完成的一个小作业，实现了一个基于NumPy的全连接神经网络来识别MNIST手写数字。通过这个项目，我学习了神经网络的基本原理、前向传播、反向传播以及各种优化方法。

在这个全连接神经网络中，各层的维度配置如下：

1. 
   输入层 ：784个神经元 (对应28×28像素的MNIST图像)
2. 
   第一个隐藏层 ：128个神经元，权重矩阵维度为 784×128
3. 
   第二个隐藏层 ：64个神经元，权重矩阵维度为 128×64
4. 
   输出层 ：10个神经元 (对应数字0-9的分类)，权重矩阵维度为 64×10

## 项目结构
```
├── main.py                     # 主程序，包含神经网络的核心实现
├── extensions.py               # 扩展功能
├── demo_extensions.py          # 扩展功能演示脚本
├── demo_correct_classification.py # 正确分类样本展示
├── demo_comparison_analysis.py # 不同参数对比分析工具
├── custom_image_prediction.py  # 可以识别自己手写的数字图像
├── unified_app.py              # 带GUI界面的整合应用
├── requirements.txt            # 需要安装的Python库
├── run_unified_app.bat         # 双击这个文件可以直接运行程序
└── README.md                   # 项目说明文档（就是你现在看的这个）
```

## 功能说明

### 基础功能
- 实现了全连接神经网络的前向传播和反向传播
- 使用交叉熵损失函数计算误差
- 支持小批量梯度下降训练
- 可以画出训练过程中的损失和准确率曲线
- 能够显示模型分类正确和错误的样例图像

### 额外功能（我自己加的）
- **多种优化算法**：支持SGD、Momentum、Adam三种优化方法
- **不同损失函数**：可以切换交叉熵和均方误差(MSE)两种损失函数
- **正则化方法**：实现了L1和L2正则化来防止过拟合
- **图形界面**：写了个简单的GUI，用起来更方便
- **自定义图像识别**：可以上传自己手写的数字让模型识别

## 怎么使用

### 第一步：安装依赖
先安装运行程序需要的Python库：
```bash
pip install -r requirements.txt
```

### 第二步：运行程序

我强烈推荐运行
python unified_app.py
直接运行各个Python文件：
最后会得到一个界面完成所有功能演示
更是这个项目的核心






```bash
# 运行主程序
python main.py

# 运行扩展功能演示
python demo_extensions.py

# 运行正确分类展示
python demo_correct_classification.py
```

程序会自动下载MNIST数据集，然后训练模型，最后输出结果和生成一些图片。

### 试试识别自己的手写数字
你可以用`custom_image_prediction.py`脚本上传自己写的数字图片来测试：
```bash
python custom_image_prediction.py
```
运行后，按照提示输入图片路径就可以了。记得把数字写大一点、清楚一点，最好是黑底白字或者白底黑字的图片。

## 神经网络的结构
我设计的神经网络结构是这样的：
- 输入层：784个神经元（因为MNIST图片是28×28的）
- 隐藏层：有两个隐藏层，分别是128和64个神经元
- 输出层：10个神经元（对应0-9这10个数字）
- 激活函数：隐藏层用的是ReLU，输出层用的是Softmax

主要参数设置：
- 学习率：0.01
- 训练轮数：20
- 批量大小：64
- 权重初始化：用的是He初始化方法

## 扩展功能怎么用
在`main.py`文件里，可以通过切换不同的神经网络类来测试各种功能：

```python
# 基本的神经网络
model = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate)

# 使用不同的优化算法（可选：'sgd', 'momentum', 'adam'）
model = NeuralNetworkWithOptimizers(input_size, hidden_sizes, output_size, learning_rate, optimizer='adam')

# 使用不同的损失函数（可选：'cross_entropy', 'mse'）
model = NeuralNetworkWithDifferentLosses(input_size, hidden_sizes, output_size, learning_rate, loss_function='cross_entropy')

# 使用不同的正则化方法（可选：'none', 'l1', 'l2'）
model = NeuralNetworkWithRegularization(input_size, hidden_sizes, output_size, learning_rate, regularization='l2', lambda_reg=0.001)
```

## 我做的实验结果
运行程序后，会生成一些图片文件：
1. `training_results.png`：显示训练过程中的损失和准确率变化
2. `misclassified_samples.png`：模型分类错误的样本
3. `correctly_classified_samples.png`：模型分类正确的样本

如果运行扩展功能，还会生成更多图表，比如不同激活函数的对比图、混淆矩阵、网络权重可视化等等。

## 遇到的问题和注意事项
1. 第一次运行的时候需要下载MNIST数据集，记得保持网络畅通
2. 训练模型可能需要一点时间，具体要看电脑性能
3. 用自己的图片测试时，最好让数字居中且占满整个图片，这样识别效果会更好
4. 如果你想改代码，可以调整学习率、隐藏层大小等参数，看看会不会让模型性能更好



## 总结
通过这个小项目，我对神经网络的工作原理有了更深刻的理解，特别是前向传播和反向传播的数学推导和代码实现。虽然模型还不够完美，但作为一个学习项目，我觉得已经达到了预期的目标。
