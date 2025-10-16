import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from extensions import (
    NeuralNetworkWithExtensions, 
    EarlyStopping, 
    LearningRateScheduler, 
    plot_confusion_matrix,
    plot_learning_rate,
    visualize_weights,
    load_cifar10
)

def preprocess_mnist():
    """加载并预处理MNIST数据集"""
    print("加载MNIST数据集...")
    # 加载MNIST数据集
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # 数据预处理
    print("预处理数据...")
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    # 转换标签为one-hot编码
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(f"数据加载完成！训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def demo_early_stopping_and_lr_scheduler():
    """演示早停策略和学习率调度器"""
    print("\n=== 演示：早停策略和学习率调度器 ===")
    
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_mnist()
    
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
    
    print("训练模型...")
    # 训练模型，使用LeakyReLU激活函数、早停策略和学习率调度器
    train_losses, val_losses, train_accuracies, val_accuracies, learning_rates = model.train(
        X_train, y_train, X_val, y_val, epochs, batch_size,
        activation='leaky_relu', early_stopping=early_stopping, lr_scheduler=lr_scheduler
    )
    
    # 评估模型
    print("在测试集上评估模型...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, activation='leaky_relu')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    # 绘制训练结果
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
    plt.savefig('early_stopping_lr_scheduler_results.png')
    plt.show()
    
    # 绘制学习率曲线
    plot_learning_rate(learning_rates)
    
    return model, X_test, y_test

def demo_activation_functions():
    """演示不同激活函数的效果"""
    print("\n=== 演示：不同激活函数的效果 ===")
    
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_mnist()
    
    # 定义超参数
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    learning_rate = 0.01
    epochs = 15
    batch_size = 64
    
    # 要测试的激活函数
    activation_functions = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu']
    
    results = {}
    
    for activation in activation_functions:
        print(f"\n使用 {activation} 激活函数训练模型...")
        
        # 创建神经网络实例
        model = NeuralNetworkWithExtensions(input_size, hidden_sizes, output_size, learning_rate)
        
        # 训练模型
        train_losses, val_losses, train_accuracies, val_accuracies = model.train(
            X_train, y_train, X_val, y_val, epochs, batch_size,
            activation=activation
        )
        
        # 评估模型
        test_loss, test_accuracy = model.evaluate(X_test, y_test, activation=activation)
        print(f'{activation} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        # 保存结果
        results[activation] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    # 绘制不同激活函数的性能比较
    print("\n绘制不同激活函数的性能比较...")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    for activation in activation_functions:
        plt.plot(results[activation]['train_losses'], label=activation)
    plt.title('Training Loss by Activation Function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for activation in activation_functions:
        plt.plot(results[activation]['val_losses'], label=activation)
    plt.title('Validation Loss by Activation Function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 3)
    for activation in activation_functions:
        plt.plot(results[activation]['train_accuracies'], label=activation)
    plt.title('Training Accuracy by Activation Function')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for activation in activation_functions:
        plt.plot(results[activation]['val_accuracies'], label=activation)
    plt.title('Validation Accuracy by Activation Function')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('activation_functions_comparison.png')
    plt.show()
    
    # 打印测试集上的最终性能
    print("\n测试集上的最终性能:")
    for activation in activation_functions:
        print(f'{activation}: Accuracy = {results[activation]["test_accuracy"]:.4f}, Loss = {results[activation]["test_loss"]:.4f}')

    return results


def demo_model_saving_and_loading():
    """演示模型保存和加载功能"""
    print("\n=== 演示：模型保存和加载功能 ===")
    
    # 加载数据
    _, _, _, _, X_test, y_test = preprocess_mnist()
    
    # 创建或加载模型
    try:
        # 尝试加载已保存的模型
        print("尝试加载已保存的模型...")
        model = NeuralNetworkWithExtensions.load_model('mnist_model.pkl')
    except:
        # 如果没有保存的模型，创建并训练一个新模型
        print("没有找到已保存的模型，创建并训练一个新模型...")
        X_train, y_train, X_val, y_val, _, _ = preprocess_mnist()
        
        # 定义超参数
        input_size = 784
        hidden_sizes = [128, 64]
        output_size = 10
        learning_rate = 0.01
        epochs = 10
        batch_size = 64
        
        # 创建神经网络实例
        model = NeuralNetworkWithExtensions(input_size, hidden_sizes, output_size, learning_rate)
        
        # 训练模型
        model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
        # 保存模型
        model.save_model('mnist_model.pkl')
    
    # 使用加载的模型进行预测
    print("使用模型进行预测...")
    y_pred = model.forward_propagation(X_test)
    
    # 评估模型
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'加载的模型 - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    return model

def demo_cifar10_support():
    """演示CIFAR-10数据集支持"""
    print("\n=== 演示：CIFAR-10数据集支持 ===")
    
    # 尝试加载CIFAR-10数据集
    result = load_cifar10()
    
    if result is not None:
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = result
        
        print(f"CIFAR-10数据加载完成！训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        print(f"CIFAR-10类别: {class_names}")
        
        # 注意：CIFAR-10图像尺寸为32x32x3=3072个特征，与MNIST的784不同
        print("\n注意：要在CIFAR-10上训练模型，需要调整输入层大小为3072")
        print("示例代码:")
        print("model = NeuralNetworkWithExtensions(input_size=3072, hidden_sizes=[256, 128], output_size=10, learning_rate=0.01)")
        print("model.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=64)")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, class_names
    else:
        print("加载CIFAR-10数据集失败，请确保tensorflow已正确安装")
        return None

def demo_enhanced_visualization(model, X_test, y_test):
    """演示增强的可视化功能"""
    print("\n=== 演示：增强的可视化功能 ===")
    
    if model is None or X_test is None or y_test is None:
        print("无法进行可视化演示，请先运行其他演示以获取模型和数据")
        return
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    y_pred = model.forward_propagation(X_test, activation='leaky_relu')
    plot_confusion_matrix(y_test, y_pred)
    
    # 可视化网络权重
    print("可视化网络权重...")
    visualize_weights(model)
    
    # 绘制错误分类的样本
    print("绘制错误分类的样本...")
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    misclassified_indices = np.where(y_pred_labels != y_true_labels)[0]
    
    if len(misclassified_indices) > 0:
        plt.figure(figsize=(10, 10))
        
        # 随机选择一些错误分类的样本
        selected_indices = np.random.choice(misclassified_indices, min(25, len(misclassified_indices)), replace=False)
        
        for i, idx in enumerate(selected_indices):
            plt.subplot(5, 5, i+1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f'T: {y_true_labels[idx]}, P: {y_pred_labels[idx]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassified_samples_extended.png')
        plt.show()
    else:
        print("没有找到错误分类的样本！")

def main():
    """主函数，演示所有扩展功能"""
    print("===== 全连接神经网络扩展功能演示 =====")
    
    # 选择要演示的功能
    print("\n请选择要演示的功能：")
    print("1. 早停策略和学习率调度器")
    print("2. 不同激活函数的效果")
    print("3. 模型保存和加载功能")
    print("4. CIFAR-10数据集支持")
    print("5. 增强的可视化功能")
    print("6. 运行所有演示")
    
    choice = input("请输入您的选择 (1-6): ")
    
    model = None
    X_test = None
    y_test = None
    
    if choice == '1':
        model, X_test, y_test = demo_early_stopping_and_lr_scheduler()
    elif choice == '2':
        demo_activation_functions()
    elif choice == '3':
        model = demo_model_saving_and_loading()
    elif choice == '4':
        demo_cifar10_support()
    elif choice == '5':
        # 如果没有模型，先加载或创建一个
        if model is None:
            try:
                model = NeuralNetworkWithExtensions.load_model('mnist_model.pkl')
                _, _, _, _, X_test, y_test = preprocess_mnist()
            except:
                print("请先运行其他演示以获取模型和数据")
                return
        demo_enhanced_visualization(model, X_test, y_test)
    elif choice == '6':
        # 运行所有演示
        model, X_test, y_test = demo_early_stopping_and_lr_scheduler()
        demo_activation_functions()
        demo_model_saving_and_loading()
        demo_cifar10_support()
        demo_enhanced_visualization(model, X_test, y_test)
    else:
        print("无效的选择，请输入1-6之间的数字")

if __name__ == "__main__":
    main()