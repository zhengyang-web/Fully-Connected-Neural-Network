import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from main import NeuralNetworkWithRegularization, plot_correctly_classified_samples

def load_trained_model():
    """加载一个预训练的神经网络模型"""
    input_size = 784  # 28*28
    hidden_sizes = [128, 64]  # 两个隐藏层，分别有128和64个神经元
    output_size = 10  # 10个类别（数字0-9）
    learning_rate = 0.01
    
    # 创建模型实例
    model = NeuralNetworkWithRegularization(
        input_size, hidden_sizes, output_size, 
        learning_rate, regularization='l2', lambda_reg=0.001
    )
    
    return model

def preprocess_test_data():
    """预处理测试数据"""
    # 加载MNIST数据集
    (_, _), (X_test, y_test) = mnist.load_data()
    
    # 将图像数据展平为一维向量
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')
    
    # 归一化数据到[0, 1]
    X_test /= 255.0
    
    # 将标签转换为one-hot编码
    encoder = OneHotEncoder(sparse_output=False)
    y_test = encoder.fit_transform(y_test.reshape(-1, 1))
    
    return X_test, y_test

def demo_correct_classification():
    """演示正确分类的样本"""
    print("=== MNIST手写数字识别 - 正确分类演示 ===")
    
    # 加载并预处理测试数据
    print("加载和预处理测试数据...")
    X_test, y_test = preprocess_test_data()
    
    # 创建模型实例
    print("初始化神经网络模型...")
    model = load_trained_model()
    
    # 注意：由于这是一个演示脚本，我们没有加载预训练的权重
    # 实际应用中，您可以在训练后保存权重，然后在这里加载
    
    # 为了快速演示，我们将使用一小部分数据
    print("使用小部分数据进行快速演示...")
    sample_indices = np.random.choice(range(len(X_test)), 100, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    # 为了展示效果，我们将手动创建一些"预测正确"的结果
    # 这只是为了演示可视化功能，不是真实的模型预测
    print("创建演示用的正确分类结果...")
    
    # 创建一个简单的模拟函数来展示效果
    def mock_predict(X):
        # 这个简单的模拟只是为了生成一些看起来合理的"正确分类"结果
        # 在实际应用中，这里应该是真实的模型预测
        y_pred = np.zeros((X.shape[0], 10))
        y_true_labels = np.argmax(y_sample, axis=1)
        
        # 80%的样本会被"正确分类"
        for i in range(X.shape[0]):
            if np.random.random() < 0.8:
                # 正确分类
                y_pred[i, y_true_labels[i]] = 1.0
            else:
                # 错误分类（随机选一个不同的类别）
                wrong_label = (y_true_labels[i] + np.random.randint(1, 9)) % 10
                y_pred[i, wrong_label] = 1.0
        
        # 添加一些随机性使结果更真实
        y_pred += np.random.normal(0, 0.1, y_pred.shape)
        # 归一化概率
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        
        return y_pred
    
    # 临时替换模型的forward_propagation方法以便演示
    original_forward = model.forward_propagation
    model.forward_propagation = mock_predict
    
    # 绘制正确分类的样本
    print("绘制正确分类的样本...")
    plot_correctly_classified_samples(model, X_sample, y_sample, 25)
    
    # 恢复原始方法
    model.forward_propagation = original_forward
    
    print("\n=== 演示完成 ===")
    print("正确分类的样本已保存为'correctly_classified_samples.png'")
    print("\n提示：")
    print("1. 如果您想查看真实训练后的模型正确分类结果，请运行'main.py'")
    print("2. 该演示使用了模拟预测结果，仅用于展示可视化效果")

if __name__ == "__main__":
    demo_correct_classification()