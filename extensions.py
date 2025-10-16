import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class EarlyStopping:
    """早停策略实现，用于防止过拟合"""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        """保存当前最佳模型"""
        # 这里我们保存模型的权重和偏置作为最佳模型
        self.best_model = {
            'weights': [w.copy() for w in model.weights],
            'biases': [b.copy() for b in model.biases]
        }

class LearningRateScheduler:
    """学习率调度器"""
    def __init__(self, initial_lr, decay_type='step', decay_rate=0.1, decay_steps=10, min_lr=1e-6):
        self.initial_lr = initial_lr
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.current_lr = initial_lr
    
    def update_lr(self, epoch):
        """根据当前轮数更新学习率"""
        if self.decay_type == 'step':
            # 阶梯式衰减
            self.current_lr = self.initial_lr * (self.decay_rate ** (epoch // self.decay_steps))
        elif self.decay_type == 'exponential':
            # 指数衰减
            self.current_lr = self.initial_lr * (self.decay_rate ** epoch)
        elif self.decay_type == 'linear':
            # 线性衰减
            self.current_lr = max(self.min_lr, self.initial_lr * (1 - epoch / 100))
        elif self.decay_type == 'cosine':
            # 余弦退火衰减
            self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * epoch / 100))
        
        # 确保学习率不低于最小值
        self.current_lr = max(self.min_lr, self.current_lr)
        
        return self.current_lr

class NeuralNetworkWithExtensions:
    """带有扩展功能的神经网络类"""
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        # 初始化权重和偏置
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.weights = []
        self.biases = []
        
        # 初始化权重和偏置（使用He初始化）
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size))
        self.biases.append(np.zeros(hidden_sizes[0]))
        
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) * np.sqrt(2.0 / hidden_sizes[i]))
            self.biases.append(np.zeros(hidden_sizes[i+1]))
        
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2.0 / hidden_sizes[-1]))
        self.biases.append(np.zeros(output_size))
        
        self.activations = []
        self.zs = []
    
    def activation_function(self, x, func_name='relu'):
        """多种激活函数的实现"""
        if func_name == 'relu':
            return np.maximum(0, x)
        elif func_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func_name == 'tanh':
            return np.tanh(x)
        elif func_name == 'leaky_relu':
            alpha = 0.01
            return np.maximum(alpha * x, x)
        elif func_name == 'elu':
            alpha = 1.0
            return np.where(x > 0, x, alpha * (np.exp(x) - 1))
        else:
            raise ValueError(f"不支持的激活函数: {func_name}")
    
    def activation_derivative(self, x, func_name='relu'):
        """激活函数的导数"""
        if func_name == 'relu':
            return (x > 0).astype(float)
        elif func_name == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)
        elif func_name == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif func_name == 'leaky_relu':
            alpha = 0.01
            dx = np.ones_like(x)
            dx[x < 0] = alpha
            return dx
        elif func_name == 'elu':
            alpha = 1.0
            return np.where(x > 0, 1, alpha * np.exp(x))
        else:
            raise ValueError(f"不支持的激活函数: {func_name}")
    
    def softmax(self, x):
        """softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止数值溢出
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_propagation(self, X, activation='relu'):
        """前向传播，支持不同的激活函数"""
        self.activations = [X]
        self.zs = []
        
        # 计算隐藏层
        for i in range(len(self.hidden_sizes)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.zs.append(z)
            a = self.activation_function(z, func_name=activation)
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
    
    def backward_propagation(self, X, y_true, activation='relu'):
        """反向传播，支持不同的激活函数"""
        m = X.shape[0]
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        
        # 输出层的误差
        delta = self.activations[-1] - y_true
        grad_weights[-1] = np.dot(self.activations[-2].T, delta) / m
        grad_biases[-1] = np.sum(delta, axis=0) / m
        
        # 隐藏层的误差
        for i in range(len(self.hidden_sizes), 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.zs[i-1], func_name=activation)
            grad_weights[i-1] = np.dot(self.activations[i-1].T, delta) / m
            grad_biases[i-1] = np.sum(delta, axis=0) / m
        
        return grad_weights, grad_biases
    
    def update_parameters(self, grad_weights, grad_biases):
        """更新参数"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, 
              activation='relu', early_stopping=None, lr_scheduler=None):
        """训练模型，支持早停策略和学习率调度器"""
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        learning_rates = []
        
        m = X_train.shape[0]
        
        for epoch in range(epochs):
            # 学习率调度
            if lr_scheduler is not None:
                self.learning_rate = lr_scheduler.update_lr(epoch)
                learning_rates.append(self.learning_rate)
            
            # 打乱训练数据
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            # 小批量梯度下降
            for i in range(0, m, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # 前向传播
                y_pred = self.forward_propagation(X_batch, activation=activation)
                
                # 反向传播
                grad_weights, grad_biases = self.backward_propagation(X_batch, y_batch, activation=activation)
                
                # 更新参数
                self.update_parameters(grad_weights, grad_biases)
            
            # 计算训练损失和准确率
            y_train_pred = self.forward_propagation(X_train, activation=activation)
            train_loss = self.compute_loss(y_train_pred, y_train)
            train_accuracy = self.compute_accuracy(y_train_pred, y_train)
            
            # 计算验证损失和准确率
            y_val_pred = self.forward_propagation(X_val, activation=activation)
            val_loss = self.compute_loss(y_val_pred, y_val)
            val_accuracy = self.compute_accuracy(y_val_pred, y_val)
            
            # 早停策略
            if early_stopping is not None:
                early_stopping(val_loss, self)
                if early_stopping.early_stop:
                    print(f"早停于第 {epoch+1} 轮")
                    # 恢复最佳模型
                    self.weights = early_stopping.best_model['weights']
                    self.biases = early_stopping.best_model['biases']
                    break
            
            # 保存损失和准确率
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # 打印训练信息
            lr_info = f", LR: {self.learning_rate:.6f}" if lr_scheduler is not None else ""
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '\
                  f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}{lr_info}')
        
        # 如果使用了学习率调度器，返回学习率历史
        if lr_scheduler is not None:
            return train_losses, val_losses, train_accuracies, val_accuracies, learning_rates
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def compute_accuracy(self, y_pred, y_true):
        """计算准确率"""
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)
    
    def evaluate(self, X_test, y_test, activation='relu'):
        """评估模型性能"""
        y_pred = self.forward_propagation(X_test, activation=activation)
        loss = self.compute_loss(y_pred, y_test)
        accuracy = self.compute_accuracy(y_pred, y_test)
        return loss, accuracy
    
    def save_model(self, file_path):
        """保存模型到文件"""
        model_dict = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'weights': self.weights,
            'biases': self.biases
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"模型已保存到 {file_path}")
    
    @classmethod
    def load_model(cls, file_path):
        """从文件加载模型"""
        with open(file_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        # 创建模型实例
        model = cls(
            input_size=model_dict['input_size'],
            hidden_sizes=model_dict['hidden_sizes'],
            output_size=model_dict['output_size']
        )
        
        # 加载权重和偏置
        model.weights = model_dict['weights']
        model.biases = model_dict['biases']
        
        print(f"模型已从 {file_path} 加载")
        return model

# 额外的可视化函数
def plot_confusion_matrix(y_true, y_pred, classes=10, figsize=(10, 8)):
    """绘制混淆矩阵"""
    # 将one-hot编码转换为标签
    y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
    y_pred_labels = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # 绘制混淆矩阵
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(classes)], 
                yticklabels=[str(i) for i in range(classes)])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 打印分类报告
    print("分类报告:")
    print(classification_report(y_true_labels, y_pred_labels))

# 绘制学习率变化曲线
def plot_learning_rate(learning_rates):
    """绘制学习率随训练轮数的变化曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates)
    plt.xlabel('训练轮数')
    plt.ylabel('学习率')
    plt.title('学习率调度')
    plt.savefig('learning_rate.png')
    plt.show()

# 可视化网络权重
def visualize_weights(model, layer_idx=0, n_weights=10):
    """可视化网络权重"""
    # 获取指定层的权重
    weights = model.weights[layer_idx]
    
    # 对于输入层，可视化权重对应的"特征图"
    if layer_idx == 0 and model.input_size == 784:  # MNIST输入层
        plt.figure(figsize=(12, 6))
        
        # 选择前n_weights个权重进行可视化
        n_weights = min(n_weights, weights.shape[1])
        
        for i in range(n_weights):
            plt.subplot(2, n_weights//2, i+1)
            # 将权重重塑为28x28的图像
            weight_image = weights[:, i].reshape(28, 28)
            plt.imshow(weight_image, cmap='viridis')
            plt.axis('off')
            plt.title(f'神经元 {i+1}')
        
        plt.tight_layout()
        plt.savefig('weights_visualization.png')
        plt.show()
    else:
        print("仅支持可视化MNIST输入层的权重")

# 使用CIFAR-10数据集
def load_cifar10():
    """加载并预处理CIFAR-10数据集"""
    try:
        from tensorflow.keras.datasets import cifar10
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.model_selection import train_test_split
        
        # 加载CIFAR-10数据集
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # 将图像数据展平为一维向量
        X_train = X_train.reshape(X_train.shape[0], -1).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')
        
        # 归一化数据到[0, 1]
        X_train /= 255.0
        X_test /= 255.0
        
        # 将标签转换为one-hot编码
        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        
        # CIFAR-10的类别名称
        class_names = ['飞机', '汽车', '鸟类', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
        
        return X_train, y_train, X_val, y_val, X_test, y_test, class_names
    except Exception as e:
        print(f"加载CIFAR-10数据集时出错: {e}")
        return None

# 主函数示例：展示扩展功能的使用
def run_with_extensions():
    """使用扩展功能训练神经网络"""
    print("加载并预处理MNIST数据...")
    from tensorflow.keras.datasets import mnist
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    
    # 加载MNIST数据集
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # 数据预处理
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # 定义超参数
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    learning_rate = 0.01
    epochs = 30
    batch_size = 64
    
    # 创建神经网络实例
    model = NeuralNetworkWithExtensions(input_size, hidden_sizes, output_size, learning_rate)
    
    # 创建早停策略和学习率调度器
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    lr_scheduler = LearningRateScheduler(initial_lr=learning_rate, decay_type='exponential', decay_rate=0.95)
    
    print("使用扩展功能训练神经网络...")
    # 训练模型，使用LeakyReLU激活函数、早停策略和学习率调度器
    train_losses, val_losses, train_accuracies, val_accuracies, learning_rates = model.train(
        X_train, y_train, X_val, y_val, epochs, batch_size,
        activation='leaky_relu', early_stopping=early_stopping, lr_scheduler=lr_scheduler
    )
    
    # 评估模型
    print("在测试集上评估模型...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, activation='leaky_relu')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    # 保存模型
    model.save_model('mnist_model.pkl')
    
    # 绘制结果
    print("绘制训练结果...")
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
    plt.savefig('training_results_with_extensions.png')
    plt.show()
    
    # 绘制学习率曲线
    plot_learning_rate(learning_rates)
    
    # 绘制混淆矩阵
    y_pred = model.forward_propagation(X_test, activation='leaky_relu')
    plot_confusion_matrix(y_test, y_pred)
    
    # 可视化网络权重
    visualize_weights(model)

if __name__ == "__main__":
    run_with_extensions()