import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入主程序中的神经网络类
try:
    from main import NeuralNetworkWithRegularization
    from main import preprocess_data
    print("成功导入神经网络类")
except ImportError as e:
    print(f"导入错误: {e}")
    # 如果导入失败，这里提供一个简化版的神经网络类用于加载权重
    class NeuralNetworkWithRegularization:
        def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, 
                     regularization='none', lambda_reg=0.01):
            self.input_size = input_size
            self.hidden_sizes = hidden_sizes
            self.output_size = output_size
            self.learning_rate = learning_rate
            self.regularization = regularization
            self.lambda_reg = lambda_reg
            self.weights = []
            self.biases = []
            self.initialize_parameters()
        
        def initialize_parameters(self):
            layer_dims = [self.input_size] + self.hidden_sizes + [self.output_size]
            for i in range(1, len(layer_dims)):
                self.weights.append(np.random.randn(layer_dims[i-1], layer_dims[i]) * 0.01)
                self.biases.append(np.zeros((1, layer_dims[i])))
        
        def relu(self, z):
            return np.maximum(0, z)
        
        def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
        
        def softmax(self, z):
            exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        def forward_propagation(self, X):
            self.activations = [X]
            self.zs = []
            
            # 隐藏层使用ReLU激活函数
            for i in range(len(self.weights) - 1):
                z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
                self.zs.append(z)
                a = self.relu(z)
                self.activations.append(a)
            
            # 输出层使用softmax激活函数
            z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
            self.zs.append(z)
            a = self.softmax(z)
            self.activations.append(a)
            
            return self.activations[-1]

# 预处理自定义图像
def preprocess_custom_image(image_path):
    """预处理用户提供的图像，使其适合神经网络输入"""
    try:
        # 打开并转换图像为灰度
        img = Image.open(image_path).convert('L')
        
        # 调整图像大小为28x28像素
        img = img.resize((28, 28), Image.LANCZOS)
        
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 反转图像颜色（MNIST数据中数字是黑色背景白色前景）
        img_array = 255 - img_array
        
        # 展平为一维向量
        img_flatten = img_array.reshape(1, -1).astype('float32')
        
        # 归一化到[0, 1]
        img_flatten /= 255.0
        
        return img_flatten, img_array
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, None

# 加载训练好的模型参数
def load_trained_model_parameters(model, model_path='model_weights.pkl'):
    """使用pickle加载训练好的模型参数"""
    try:
        # 检查文件是否存在
        if os.path.exists(model_path):
            print(f"从{model_path}加载模型参数...")
            import pickle
            with open(model_path, 'rb') as f:
                params = pickle.load(f)
                model.weights = params['weights']
                model.biases = params['biases']
            return True
        else:
            print("未找到保存的模型参数文件。")
            print("请先运行main.py训练模型并保存参数。")
            return False
    except Exception as e:
        print(f"加载模型参数时出错: {e}")
        return False

# 创建一个可以预测自定义图像的函数
def predict_custom_image(model, image_path):
    """预测用户提供的图像中的数字"""
    # 预处理图像
    img_flatten, img_array = preprocess_custom_image(image_path)
    if img_flatten is None:
        return None
    
    # 进行预测
    prediction = model.forward_propagation(img_flatten)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # 显示结果
    plt.figure(figsize=(6, 6))
    plt.imshow(img_array, cmap='gray')
    plt.title(f'预测数字: {predicted_digit} (置信度: {confidence:.2f})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"预测结果: 数字 {predicted_digit}, 置信度: {confidence:.4f}")
    return predicted_digit

# 主函数
def main():
    print("自定义数字图像识别工具")
    print("=====================")
    
    # 创建神经网络模型
    input_size = 784  # 28*28
    hidden_sizes = [128, 64]  # 两个隐藏层，分别有128和64个神经元
    output_size = 10  # 10个类别（数字0-9）
    
    model = NeuralNetworkWithRegularization(
        input_size, hidden_sizes, output_size,
        learning_rate=0.01, regularization='l2', lambda_reg=0.001
    )
    
    # 尝试加载训练好的模型参数
    if not load_trained_model_parameters(model, 'model_weights.pkl'):
        # 如果没有保存的参数，询问用户是否要训练新模型
        train_new = input("是否要训练新模型？(y/n): ").lower()
        if train_new == 'y':
            print("正在加载MNIST数据并训练模型...")
            # 加载MNIST数据
            X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data()
            
            # 训练模型
            epochs = 20
            batch_size = 64
            print(f"开始训练，共{epochs}个epoch，批量大小{batch_size}...")
            
            try:
                # 尝试调用完整的train方法
                model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
            except AttributeError:
                # 如果是简化版类，使用简单训练
                print("使用简化版训练...")
                # 这里是简化版的训练逻辑
                for epoch in range(epochs):
                    # 简单的批量训练
                    for i in range(0, X_train.shape[0], batch_size):
                        end = min(i + batch_size, X_train.shape[0])
                        X_batch = X_train[i:end]
                        y_batch = y_train[i:end]
                        
                        # 前向传播
                        y_pred = model.forward_propagation(X_batch)
                        
                        # 简化的反向传播（这里只是为了演示）
                        # 在实际应用中，应该使用完整的反向传播算法
                        pass
                    
                    # 评估模型
                    train_pred = model.forward_propagation(X_train)
                    train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
                    print(f"Epoch {epoch+1}/{epochs}, 训练准确率: {train_acc:.4f}")
            
            # 使用pickle保存模型参数（更适合保存形状不一致的数组列表）
            import pickle
            model_params = {'weights': model.weights, 'biases': model.biases}
            with open('model_weights.pkl', 'wb') as f:
                pickle.dump(model_params, f)
            print("模型参数已保存到model_weights.pkl")
        else:
            print("程序已退出。请先训练模型或提供已训练好的模型参数文件。")
            return
    
    # 获取用户输入的图像路径
    while True:
        image_path = input("请输入要识别的图像路径（或输入'q'退出）: ")
        if image_path.lower() == 'q':
            break
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误：找不到文件 '{image_path}'")
            continue
        
        # 预测图像
        predict_custom_image(model, image_path)

if __name__ == "__main__":
    main()