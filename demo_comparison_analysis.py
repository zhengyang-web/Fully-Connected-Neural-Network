import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time

# 导入项目中的神经网络类
from main import (
    NeuralNetwork,
    NeuralNetworkWithOptimizers,
    NeuralNetworkWithDifferentLosses,
    NeuralNetworkWithRegularization,
    preprocess_data,
    plot_results
)

class ExperimentManager:
    """实验管理器，用于比较不同配置下的模型性能"""
    def __init__(self):
        # 加载并预处理数据
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = preprocess_data()
        
        # 基本超参数
        self.input_size = 784  # 28*28
        self.hidden_sizes = [128, 64]  # 两个隐藏层，分别有128和64个神经元
        self.output_size = 10  # 10个类别（数字0-9）
        self.learning_rate = 0.01
        self.epochs = 20
        self.batch_size = 64
        
        # 存储实验结果
        self.experiments = {}
    
    def run_experiment(self, name, model_class, **model_kwargs):
        """运行单个实验"""
        print(f"\n=== 运行实验: {name} ===")
        
        # 创建模型实例
        model = model_class(
            self.input_size, 
            self.hidden_sizes, 
            self.output_size, 
            self.learning_rate, 
            **model_kwargs
        )
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        train_losses, val_losses, train_accuracies, val_accuracies = model.train(
            self.X_train, self.y_train, self.X_val, self.y_val, 
            self.epochs, self.batch_size
        )
        
        # 记录训练结束时间
        train_time = time.time() - start_time
        
        # 在测试集上评估模型
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test)
        
        print(f'实验: {name}')
        print(f'训练时间: {train_time:.2f}秒')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        # 保存实验结果
        self.experiments[name] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'train_time': train_time
        }
        
        return model
    
    def compare_optimizers(self):
        """比较不同优化算法的效果"""
        print("\n========== 比较不同优化算法的效果 ==========")
        
        # 定义要比较的优化算法
        optimizers = ['sgd', 'momentum', 'adam']
        
        for optimizer in optimizers:
            self.run_experiment(
                f'优化器: {optimizer}',
                NeuralNetworkWithOptimizers,
                optimizer=optimizer
            )
        
        # 可视化比较结果
        self._visualize_comparison('优化算法')
        
        # 分析结果
        self._analyze_optimizer_results()
    
    def compare_loss_functions(self):
        """比较不同损失函数的效果"""
        print("\n========== 比较不同损失函数的效果 ==========")
        
        # 定义要比较的损失函数
        loss_functions = ['cross_entropy', 'mse']
        
        for loss_function in loss_functions:
            self.run_experiment(
                f'损失函数: {loss_function}',
                NeuralNetworkWithDifferentLosses,
                loss_function=loss_function
            )
        
        # 可视化比较结果
        self._visualize_comparison('损失函数')
        
        # 分析结果
        self._analyze_loss_function_results()
    
    def compare_regularization(self):
        """比较不同正则化方法的效果"""
        print("\n========== 比较不同正则化方法的效果 ==========")
        
        # 运行基准模型（无正则化）
        self.run_experiment(
            '正则化: none',
            NeuralNetwork
        )
        
        # 定义要比较的正则化方法和参数
        regularizations = [
            ('l1', 0.01),
            ('l1', 0.001),
            ('l2', 0.01),
            ('l2', 0.001)
        ]
        
        for reg_type, reg_lambda in regularizations:
            self.run_experiment(
                f'正则化: {reg_type}, lambda={reg_lambda}',
                NeuralNetworkWithRegularization,
                regularization=reg_type,
                lambda_reg=reg_lambda
            )
        
        # 可视化比较结果
        self._visualize_comparison('正则化方法')
        
        # 分析结果
        self._analyze_regularization_results()
    
    def _visualize_comparison(self, title_suffix):
        """可视化比较结果"""
        # 创建一个大图，包含4个子图
        plt.figure(figsize=(14, 12))
        
        # 训练损失曲线
        plt.subplot(2, 2, 1)
        for name, result in self.experiments.items():
            plt.plot(result['train_losses'], label=name)
        plt.title(f'训练损失比较 - {title_suffix}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 验证损失曲线
        plt.subplot(2, 2, 2)
        for name, result in self.experiments.items():
            plt.plot(result['val_losses'], label=name)
        plt.title(f'验证损失比较 - {title_suffix}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 训练准确率曲线
        plt.subplot(2, 2, 3)
        for name, result in self.experiments.items():
            plt.plot(result['train_accuracies'], label=name)
        plt.title(f'训练准确率比较 - {title_suffix}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 验证准确率曲线
        plt.subplot(2, 2, 4)
        for name, result in self.experiments.items():
            plt.plot(result['val_accuracies'], label=name)
        plt.title(f'验证准确率比较 - {title_suffix}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{title_suffix}_comparison.png')
        plt.show()
        
        # 创建一个测试性能对比图
        plt.figure(figsize=(10, 6))
        
        # 提取测试准确率和训练时间
        names = list(self.experiments.keys())
        test_accuracies = [result['test_accuracy'] for result in self.experiments.values()]
        train_times = [result['train_time'] for result in self.experiments.values()]
        
        # 绘制测试准确率对比
        plt.subplot(1, 2, 1)
        plt.bar(names, test_accuracies, color='skyblue')
        plt.title(f'测试准确率对比 - {title_suffix}')
        plt.xlabel('配置')
        plt.ylabel('准确率')
        plt.ylim(min(test_accuracies) - 0.02, max(test_accuracies) + 0.02)  # 设置y轴范围，使差异更明显
        plt.xticks(rotation=45, ha='right')
        
        # 绘制训练时间对比
        plt.subplot(1, 2, 2)
        plt.bar(names, train_times, color='lightgreen')
        plt.title(f'训练时间对比 - {title_suffix}')
        plt.xlabel('配置')
        plt.ylabel('时间(秒)')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{title_suffix}_test_performance.png')
        plt.show()
    
    def _analyze_optimizer_results(self):
        """分析不同优化算法的结果"""
        print("\n========== 优化算法效果分析 ==========")
        
        # 提取结果
        results = {}
        for name, result in self.experiments.items():
            optimizer = name.split(': ')[1]
            results[optimizer] = result
        
        # 分析收敛速度
        print("\n1. 收敛速度分析:")
        for optimizer in ['sgd', 'momentum', 'adam']:
            # 找到第一个训练损失低于某个阈值的轮次
            threshold = 0.2
            conv_epoch = next((i for i, loss in enumerate(results[optimizer]['train_losses']) if loss < threshold), self.epochs)
            print(f"   {optimizer} 达到损失 < {threshold} 的轮次: {conv_epoch+1}/{self.epochs}")
        
        # 分析最终性能
        print("\n2. 最终性能分析:")
        sorted_by_accuracy = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        for i, (optimizer, result) in enumerate(sorted_by_accuracy):
            print(f"   排名 {i+1}: {optimizer} - 测试准确率: {result['test_accuracy']:.4f}, 训练时间: {result['train_time']:.2f}秒")
        
        # 分析优缺点
        print("\n3. 各优化算法优缺点分析:")
        print("   - SGD: 实现简单，但收敛速度慢，容易陷入局部最优")
        print("   - Momentum: 通过动量加速收敛，对噪声更鲁棒")
        print("   - Adam: 结合了动量和自适应学习率，收敛速度快，通常性能最好")
        
        print("\n结论: 在大多数情况下，Adam优化算法提供了最佳的收敛速度和最终性能。")
    
    def _analyze_loss_function_results(self):
        """分析不同损失函数的结果"""
        print("\n========== 损失函数效果分析 ==========")
        
        # 提取结果
        results = {}
        for name, result in self.experiments.items():
            loss = name.split(': ')[1]
            results[loss] = result
        
        # 分析收敛行为
        print("\n1. 收敛行为分析:")
        for loss in ['cross_entropy', 'mse']:
            # 计算损失下降幅度
            initial_loss = results[loss]['train_losses'][0]
            final_loss = results[loss]['train_losses'][-1]
            reduction = (initial_loss - final_loss) / initial_loss * 100
            print(f"   {loss} 损失下降幅度: {reduction:.2f}%")
        
        # 分析最终性能
        print("\n2. 最终性能分析:")
        sorted_by_accuracy = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        for i, (loss, result) in enumerate(sorted_by_accuracy):
            print(f"   排名 {i+1}: {loss} - 测试准确率: {result['test_accuracy']:.4f}, 训练时间: {result['train_time']:.2f}秒")
        
        # 分析优缺点
        print("\n3. 各损失函数优缺点分析:")
        print("   - cross_entropy: 对于分类问题更自然，收敛速度快，梯度信号强")
        print("   - mse: 对于回归问题更合适，但在分类问题中可能导致训练不稳定")
        
        print("\n结论: 对于MNIST分类任务，交叉熵损失函数通常比均方误差表现更好。")
    
    def _analyze_regularization_results(self):
        """分析不同正则化方法的结果"""
        print("\n========== 正则化效果分析 ==========")
        
        # 提取结果
        results = {}
        for name, result in self.experiments.items():
            results[name] = result
        
        # 分析过拟合情况
        print("\n1. 过拟合情况分析:")
        for name, result in results.items():
            # 计算训练准确率和验证准确率之间的差距
            train_acc = result['train_accuracies'][-1]
            val_acc = result['val_accuracies'][-1]
            gap = train_acc - val_acc
            print(f"   {name} - 训练/验证准确率差距: {gap:.4f}")
        
        # 分析最终性能
        print("\n2. 最终性能分析:")
        sorted_by_accuracy = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        for i, (name, result) in enumerate(sorted_by_accuracy):
            print(f"   排名 {i+1}: {name} - 测试准确率: {result['test_accuracy']:.4f}")
        
        # 分析正则化强度影响
        print("\n3. 正则化强度影响分析:")
        # 找出L1正则化的结果
        l1_results = {name: result for name, result in results.items() if 'l1' in name}
        if l1_results:
            print("   L1正则化:")
            for name in sorted(l1_results.keys()):
                print(f"     {name}: 测试准确率 = {l1_results[name]['test_accuracy']:.4f}")
        
        # 找出L2正则化的结果
        l2_results = {name: result for name, result in results.items() if 'l2' in name}
        if l2_results:
            print("   L2正则化:")
            for name in sorted(l2_results.keys()):
                print(f"     {name}: 测试准确率 = {l2_results[name]['test_accuracy']:.4f}")
        
        print("\n结论: 适当的正则化可以减少过拟合，提高模型的泛化能力。通常，较小的正则化系数（如lambda=0.001）效果更好。")
    
    def run_all_comparisons(self):
        """运行所有比较实验"""
        print("===== 全连接神经网络参数效果比较实验 =====")
        
        # 比较不同优化算法
        self.experiments = {}
        self.compare_optimizers()
        
        # 比较不同损失函数
        self.experiments = {}
        self.compare_loss_functions()
        
        # 比较不同正则化方法
        self.experiments = {}
        self.compare_regularization()
        
        print("\n===== 所有比较实验已完成 =====")
        print("实验结果图表已保存到当前目录。")


def main():
    """主函数"""
    # 创建实验管理器
    manager = ExperimentManager()
    
    # 打印菜单
    print("===== 全连接神经网络参数效果比较工具 =====")
    print("此工具用于比较不同参数设置对MNIST分类任务的影响")
    print("\n请选择要运行的比较实验:")
    print("1. 比较不同优化算法 (SGD, Momentum, Adam)")
    print("2. 比较不同损失函数 (交叉熵, 均方误差)")
    print("3. 比较不同正则化方法 (无正则化, L1, L2)")
    print("4. 运行所有比较实验")
    
    # 获取用户选择
    choice = input("请输入您的选择 (1-4): ")
    
    # 运行相应的比较实验
    if choice == '1':
        manager.compare_optimizers()
    elif choice == '2':
        manager.compare_loss_functions()
    elif choice == '3':
        manager.compare_regularization()
    elif choice == '4':
        manager.run_all_comparisons()
    else:
        print("无效的选择，请输入1-4之间的数字")


if __name__ == "__main__":
    main()