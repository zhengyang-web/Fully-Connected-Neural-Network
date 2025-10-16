import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        # 输入层到第一个隐藏层
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size))
        self.biases.append(np.zeros(hidden_sizes[0]))
        
        # 隐藏层之间的连接
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) * np.sqrt(2.0 / hidden_sizes[i]))
            self.biases.append(np.zeros(hidden_sizes[i+1]))
        
        # 最后一个隐藏层到输出层
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2.0 / hidden_sizes[-1]))
        self.biases.append(np.zeros(output_size))
        
        # 用于存储激活值和加权输入，以便反向传播使用
        self.activations = []
        self.zs = []
    
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU激活函数的导数"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止数值溢出
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """前向传播"""
        self.activations = [X]
        self.zs = []
        
        # 计算隐藏层
        for i in range(len(self.hidden_sizes)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.zs.append(z)
            a = self.relu(z)
            self.activations.append(a)
        
        # 计算输出层
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.zs.append(z)
        a = self.softmax(z)
        self.activations.append(a)
        
        return self.activations[-1]
    
    def compute_loss(self, y_pred, y_true):
        """计算交叉熵损失"""
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-10)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward_propagation(self, X, y_true):
        """反向传播"""
        m = X.shape[0]
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        
        # 输出层的误差
        delta = self.activations[-1] - y_true
        grad_weights[-1] = np.dot(self.activations[-2].T, delta) / m
        grad_biases[-1] = np.sum(delta, axis=0) / m
        
        # 隐藏层的误差
        for i in range(len(self.hidden_sizes), 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.zs[i-1])
            grad_weights[i-1] = np.dot(self.activations[i-1].T, delta) / m
            grad_biases[i-1] = np.sum(delta, axis=0) / m
        
        return grad_weights, grad_biases
    
    def update_parameters(self, grad_weights, grad_biases):
        """更新参数"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """训练模型"""
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        m = X_train.shape[0]
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 打乱训练数据
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            # 小批量梯度下降
            for i in range(0, m, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # 前向传播
                y_pred = self.forward_propagation(X_batch)
                
                # 反向传播
                grad_weights, grad_biases = self.backward_propagation(X_batch, y_batch)
                
                # 更新参数
                self.update_parameters(grad_weights, grad_biases)
            
            # 计算训练损失和准确率
            y_train_pred = self.forward_propagation(X_train)
            train_loss = self.compute_loss(y_train_pred, y_train)
            train_accuracy = self.compute_accuracy(y_train_pred, y_train)
            
            # 计算验证损失和准确率
            y_val_pred = self.forward_propagation(X_val)
            val_loss = self.compute_loss(y_val_pred, y_val)
            val_accuracy = self.compute_accuracy(y_val_pred, y_val)
            
            # 保存损失和准确率
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # 打印训练信息
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '\
                  f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Time: {epoch_time:.2f}s')
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def compute_accuracy(self, y_pred, y_true):
        """计算准确率"""
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.forward_propagation(X_test)
        loss = self.compute_loss(y_pred, y_test)
        accuracy = self.compute_accuracy(y_pred, y_test)
        return loss, accuracy
    
    def predict(self, X):
        """预测类别"""
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)

# 附加：不同优化算法的实现
class NeuralNetworkWithOptimizers(NeuralNetwork):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, optimizer='sgd', beta=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(input_size, hidden_sizes, output_size, learning_rate)
        self.optimizer = optimizer
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # 初始化时间步，所有优化器都需要
        
        # 初始化优化器参数
        if optimizer in ['momentum', 'adam']:
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
        
        if optimizer == 'adam':
            self.s_weights = [np.zeros_like(w) for w in self.weights]
            self.s_biases = [np.zeros_like(b) for b in self.biases]
    
    def update_parameters(self, grad_weights, grad_biases):
        """使用不同的优化算法更新参数"""
        self.t += 1  # 用于Adam优化器的时间步
        
        if self.optimizer == 'sgd':
            # 随机梯度下降
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grad_weights[i]
                self.biases[i] -= self.learning_rate * grad_biases[i]
        
        elif self.optimizer == 'momentum':
            # Momentum优化器
            for i in range(len(self.weights)):
                self.v_weights[i] = self.beta * self.v_weights[i] + (1 - self.beta) * grad_weights[i]
                self.v_biases[i] = self.beta * self.v_biases[i] + (1 - self.beta) * grad_biases[i]
                self.weights[i] -= self.learning_rate * self.v_weights[i]
                self.biases[i] -= self.learning_rate * self.v_biases[i]
        
        elif self.optimizer == 'adam':
            # Adam优化器
            for i in range(len(self.weights)):
                # 更新一阶矩估计
                self.v_weights[i] = self.beta * self.v_weights[i] + (1 - self.beta) * grad_weights[i]
                self.v_biases[i] = self.beta * self.v_biases[i] + (1 - self.beta) * grad_biases[i]
                
                # 更新二阶矩估计
                self.s_weights[i] = self.beta2 * self.s_weights[i] + (1 - self.beta2) * (grad_weights[i] ** 2)
                self.s_biases[i] = self.beta2 * self.s_biases[i] + (1 - self.beta2) * (grad_biases[i] ** 2)
                
                # 偏差修正
                v_weights_corrected = self.v_weights[i] / (1 - self.beta ** self.t)
                v_biases_corrected = self.v_biases[i] / (1 - self.beta ** self.t)
                s_weights_corrected = self.s_weights[i] / (1 - self.beta2 ** self.t)
                s_biases_corrected = self.s_biases[i] / (1 - self.beta2 ** self.t)
                
                # 更新参数
                self.weights[i] -= self.learning_rate * v_weights_corrected / (np.sqrt(s_weights_corrected) + self.epsilon)
                self.biases[i] -= self.learning_rate * v_biases_corrected / (np.sqrt(s_biases_corrected) + self.epsilon)

# 附加：不同损失函数的实现
class NeuralNetworkWithDifferentLosses(NeuralNetwork):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, loss_function='cross_entropy'):
        super().__init__(input_size, hidden_sizes, output_size, learning_rate)
        self.loss_function = loss_function
        
    def compute_loss(self, y_pred, y_true):
        """计算不同类型的损失函数"""
        if self.loss_function == 'cross_entropy':
            # 交叉熵损失
            m = y_true.shape[0]
            log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-10)
            loss = np.sum(log_likelihood) / m
            return loss
        
        elif self.loss_function == 'mse':
            # 均方误差
            return np.mean(np.square(y_pred - y_true))
    
    def backward_propagation(self, X, y_true):
        """根据不同的损失函数调整反向传播"""
        m = X.shape[0]
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        
        if self.loss_function == 'cross_entropy':
            # 交叉熵损失的反向传播
            delta = self.activations[-1] - y_true
        
        elif self.loss_function == 'mse':
            # 均方误差的反向传播
            delta = 2 * (self.activations[-1] - y_true) * self.activations[-1] * (1 - self.activations[-1])
        
        # 其余部分的反向传播与原实现相同
        grad_weights[-1] = np.dot(self.activations[-2].T, delta) / m
        grad_biases[-1] = np.sum(delta, axis=0) / m
        
        for i in range(len(self.hidden_sizes), 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.zs[i-1])
            grad_weights[i-1] = np.dot(self.activations[i-1].T, delta) / m
            grad_biases[i-1] = np.sum(delta, axis=0) / m
        
        return grad_weights, grad_biases

# 附加：正则化方法
class NeuralNetworkWithRegularization(NeuralNetwork):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, regularization='none', lambda_reg=0.01):
        super().__init__(input_size, hidden_sizes, output_size, learning_rate)
        self.regularization = regularization
        self.lambda_reg = lambda_reg
    
    def compute_loss(self, y_pred, y_true):
        """添加正则化项到损失函数"""
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-10)
        loss = np.sum(log_likelihood) / m
        
        # 添加正则化项
        if self.regularization == 'l2':
            # L2正则化
            l2_reg = 0
            for w in self.weights:
                l2_reg += np.sum(w ** 2)
            loss += (self.lambda_reg / (2 * m)) * l2_reg
        
        elif self.regularization == 'l1':
            # L1正则化
            l1_reg = 0
            for w in self.weights:
                l1_reg += np.sum(np.abs(w))
            loss += (self.lambda_reg / m) * l1_reg
        
        return loss
    
    def backward_propagation(self, X, y_true):
        """在反向传播中添加正则化的梯度"""
        m = X.shape[0]
        grad_weights, grad_biases = super().backward_propagation(X, y_true)
        
        # 添加正则化的梯度
        if self.regularization == 'l2':
            # L2正则化的梯度
            for i in range(len(grad_weights)):
                grad_weights[i] += (self.lambda_reg / m) * self.weights[i]
        
        elif self.regularization == 'l1':
            # L1正则化的梯度
            for i in range(len(grad_weights)):
                grad_weights[i] += (self.lambda_reg / m) * np.sign(self.weights[i])
        
        return grad_weights, grad_biases

# 数据预处理函数
def preprocess_data():
    # 加载MNIST数据集
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # 将图像数据展平为一维向量
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')
    
    # 归一化数据到[0, 1]
    X_train /= 255.0
    X_test /= 255.0
    
    # 将标签转换为one-hot编码
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# 绘制结果函数
def plot_results(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

# 绘制错误分类的样本
def plot_misclassified_samples(model, X_test, y_test, num_samples=10):
    y_pred = model.forward_propagation(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    
    misclassified_indices = np.where(y_pred_labels != y_true_labels)[0]
    
    if len(misclassified_indices) > 0:
        plt.figure(figsize=(10, 10))
        
        # 随机选择一些错误分类的样本
        selected_indices = np.random.choice(misclassified_indices, min(num_samples, len(misclassified_indices)), replace=False)
        
        for i, idx in enumerate(selected_indices):
            plt.subplot(5, 5, i+1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f'True: {y_true_labels[idx]}, Pred: {y_pred_labels[idx]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassified_samples.png')
        plt.show()
    else:
        print("No misclassified samples found!")

# 绘制正确分类的样本
def plot_correctly_classified_samples(model, X_test, y_test, num_samples=10):
    y_pred = model.forward_propagation(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    
    correctly_classified_indices = np.where(y_pred_labels == y_true_labels)[0]
    
    if len(correctly_classified_indices) > 0:
        plt.figure(figsize=(10, 10))
        
        # 随机选择一些正确分类的样本
        selected_indices = np.random.choice(correctly_classified_indices, min(num_samples, len(correctly_classified_indices)), replace=False)
        
        for i, idx in enumerate(selected_indices):
            plt.subplot(5, 5, i+1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f'True: {y_true_labels[idx]}, Pred: {y_pred_labels[idx]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('correctly_classified_samples.png')
        plt.show()
    else:
        print("No correctly classified samples found!")

# 保存模型参数
def save_model_parameters(model, model_path='model_weights.pkl'):
    """使用pickle保存训练好的模型权重和偏置"""
    import pickle
    import os
    
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    
    # 使用pickle保存模型参数（更适合保存形状不一致的数组列表）
    model_params = {'weights': model.weights, 'biases': model.biases}
    with open(model_path, 'wb') as f:
        pickle.dump(model_params, f)
    print(f"模型参数已保存到{model_path}")

# 主函数
def main():
    print("Loading and preprocessing MNIST data...")
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data()
    
    # 定义超参数
    input_size = 784  # 28*28
    hidden_sizes = [128, 64]  # 两个隐藏层，分别有128和64个神经元
    output_size = 10  # 10个类别（数字0-9）
    learning_rate = 0.01
    epochs = 20
    batch_size = 64
    
    print("Initializing neural network...")
    # 创建神经网络实例
    # 可以切换到不同的神经网络类来测试不同的功能
    # model = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate)
    # model = NeuralNetworkWithOptimizers(input_size, hidden_sizes, output_size, learning_rate, optimizer='adam')
    # model = NeuralNetworkWithDifferentLosses(input_size, hidden_sizes, output_size, learning_rate, loss_function='cross_entropy')
    model = NeuralNetworkWithRegularization(input_size, hidden_sizes, output_size, learning_rate, regularization='l2', lambda_reg=0.001)
    
    print("Training neural network...")
    # 训练模型
    train_losses, val_losses, train_accuracies, val_accuracies = model.train(
        X_train, y_train, X_val, y_val, epochs, batch_size
    )
    
    print("Evaluating model on test data...")
    # 在测试集上评估模型
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    # 保存模型参数
    print("Saving model parameters...")
    save_model_parameters(model, 'model_weights.pkl')
    
    # 绘制训练和验证的损失和准确率曲线
    print("Plotting training results...")
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 绘制错误分类的样本
    print("Plotting misclassified samples...")
    plot_misclassified_samples(model, X_test, y_test)
    
    # 绘制正确分类的样本
    print("Plotting correctly classified samples...")
    plot_correctly_classified_samples(model, X_test, y_test, 25)

if __name__ == "__main__":
    main()