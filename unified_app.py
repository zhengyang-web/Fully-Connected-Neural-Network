import numpy as np
import matplotlib.pyplot as plt
# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
import os
import sys
import pickle
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from main import NeuralNetwork, NeuralNetworkWithOptimizers, NeuralNetworkWithDifferentLosses, NeuralNetworkWithRegularization
from main import preprocess_data, plot_results, plot_misclassified_samples, plot_correctly_classified_samples, save_model_parameters
from custom_image_prediction import load_trained_model_parameters
from custom_image_prediction import preprocess_custom_image, predict_custom_image
from extensions import NeuralNetworkWithExtensions, EarlyStopping, LearningRateScheduler

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class UnifiedNeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST数字识别系统")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 初始化变量
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.train_losses = None
        self.val_losses = None
        self.train_accuracies = None
        self.val_accuracies = None
        
        # 定义神经网络的超参数
        self.input_size = 784  # 28*28
        self.hidden_sizes = [128, 64]  # 两个隐藏层，分别有128和64个神经元
        self.output_size = 10  # 10个类别（数字0-9）
        self.learning_rate = 0.01
        self.epochs = 20
        self.batch_size = 64
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 创建右侧显示面板
        self.display_frame = ttk.LabelFrame(self.main_frame, text="结果显示")
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 创建进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # 创建日志文本框
        self.log_frame = ttk.LabelFrame(self.main_frame, text="日志")
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=5, pady=5)
        self.log_text = tk.Text(self.log_frame, height=5, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        self.log_text.config(state=tk.DISABLED)
        
        # 创建按钮
        self.create_buttons()
        
    def create_buttons(self):
        """创建所有功能按钮"""
        # 数据管理按钮组
        data_group = ttk.LabelFrame(self.control_frame, text="数据管理")
        data_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(data_group, text="加载MNIST数据集", command=self.load_data).pack(fill=tk.X, padx=5, pady=5)
        
        # 模型管理按钮组
        model_group = ttk.LabelFrame(self.control_frame, text="模型管理")
        model_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(model_group, text="创建基本模型", command=lambda: self.create_model('basic')).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(model_group, text="创建带优化器的模型", command=lambda: self.create_model('optimizers')).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(model_group, text="创建带不同损失函数的模型", command=lambda: self.create_model('losses')).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(model_group, text="创建带正则化的模型", command=lambda: self.create_model('regularization')).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(model_group, text="保存模型", command=self.save_model).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(model_group, text="加载模型", command=self.load_model).pack(fill=tk.X, padx=5, pady=2)
        
        # 训练按钮组
        train_group = ttk.LabelFrame(self.control_frame, text="模型训练")
        train_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(train_group, text="训练当前模型", command=self.train_model).pack(fill=tk.X, padx=5, pady=5)
        
        # 预测评估按钮组
        eval_group = ttk.LabelFrame(self.control_frame, text="预测与评估")
        eval_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(eval_group, text="评估模型性能", command=self.evaluate_model).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(eval_group, text="数字图像识别", command=self.recognize_digit).pack(fill=tk.X, padx=5, pady=2)
        
        # 可视化按钮组
        viz_group = ttk.LabelFrame(self.control_frame, text="可视化")
        viz_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(viz_group, text="显示训练结果", command=self.show_training_results).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(viz_group, text="显示混淆矩阵", command=self.show_confusion_matrix).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(viz_group, text="可视化网络权重", command=self.visualize_network_weights).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(viz_group, text="显示错误分类样本", command=self.show_misclassified_samples).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(viz_group, text="显示正确分类样本", command=self.show_correctly_classified_samples).pack(fill=tk.X, padx=5, pady=2)
        
        # 扩展功能按钮组
        ext_group = ttk.LabelFrame(self.control_frame, text="扩展功能")
        ext_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(ext_group, text="早停策略和学习率调度器", command=self.demo_early_stopping_and_lr_scheduler).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(ext_group, text="比较不同激活函数", command=self.compare_activation_functions).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(ext_group, text="参数比较分析工具", command=self.run_parameter_comparison).pack(fill=tk.X, padx=5, pady=2)
        
        # 退出按钮
        ttk.Button(self.control_frame, text="退出", command=self.root.quit).pack(fill=tk.X, padx=5, pady=5)
        
    def log_message(self, message):
        """在日志文本框中显示消息"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def load_data(self):
        """加载并预处理MNIST数据集"""
        self.log_message("正在加载和预处理MNIST数据...")
        try:
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = preprocess_data()
            self.log_message("数据加载完成！")
            messagebox.showinfo("成功", "MNIST数据集加载完成！")
        except Exception as e:
            self.log_message(f"加载数据时出错: {e}")
            messagebox.showerror("错误", f"加载数据时出错: {e}")
            
    def create_model(self, model_type='basic'):
        """创建神经网络模型"""
        self.log_message(f"正在初始化{model_type}神经网络...")
        
        try:
            if model_type == 'basic':
                self.model = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size, self.learning_rate)
            elif model_type == 'optimizers':
                # 创建选择优化器的对话框
                optimizer_window = tk.Toplevel(self.root)
                optimizer_window.title("选择优化器")
                optimizer_window.geometry("300x200")
                optimizer_window.resizable(False, False)
                optimizer_window.grab_set()  # 模态对话框
                
                ttk.Label(optimizer_window, text="请选择优化算法:\n", font=("SimHei", 10)).pack(pady=10)
                
                optimizer_var = tk.StringVar(value="adam")
                
                ttk.Radiobutton(optimizer_window, text="随机梯度下降 (SGD)", variable=optimizer_var, value="sgd").pack(anchor=tk.W, padx=20)
                ttk.Radiobutton(optimizer_window, text="动量优化器 (Momentum)", variable=optimizer_var, value="momentum").pack(anchor=tk.W, padx=20)
                ttk.Radiobutton(optimizer_window, text="Adam优化器", variable=optimizer_var, value="adam").pack(anchor=tk.W, padx=20)
                
                def create_optimizer_model():
                    optimizer = optimizer_var.get()
                    self.model = NeuralNetworkWithOptimizers(
                        self.input_size, self.hidden_sizes, self.output_size, 
                        learning_rate=self.learning_rate, optimizer=optimizer
                    )
                    optimizer_window.destroy()
                    self.log_message(f"带{optimizer}优化器的神经网络初始化完成！")
                    messagebox.showinfo("成功", f"带{optimizer}优化器的神经网络初始化完成！")
                
                ttk.Button(optimizer_window, text="确定", command=create_optimizer_model).pack(pady=10)
                return  # 等待用户选择后再继续
            elif model_type == 'losses':
                # 创建选择损失函数的对话框
                loss_window = tk.Toplevel(self.root)
                loss_window.title("选择损失函数")
                loss_window.geometry("300x200")
                loss_window.resizable(False, False)
                loss_window.grab_set()  # 模态对话框
                
                ttk.Label(loss_window, text="请选择损失函数:\n", font=("SimHei", 10)).pack(pady=10)
                
                loss_var = tk.StringVar(value="cross_entropy")
                
                ttk.Radiobutton(loss_window, text="交叉熵损失 (Cross Entropy)", variable=loss_var, value="cross_entropy").pack(anchor=tk.W, padx=20)
                ttk.Radiobutton(loss_window, text="均方误差 (MSE)", variable=loss_var, value="mse").pack(anchor=tk.W, padx=20)
                
                def create_loss_model():
                    loss_function = loss_var.get()
                    self.model = NeuralNetworkWithDifferentLosses(
                        self.input_size, self.hidden_sizes, self.output_size, 
                        learning_rate=self.learning_rate, loss_function=loss_function
                    )
                    loss_window.destroy()
                    self.log_message(f"带{loss_function}损失函数的神经网络初始化完成！")
                    messagebox.showinfo("成功", f"带{loss_function}损失函数的神经网络初始化完成！")
                
                ttk.Button(loss_window, text="确定", command=create_loss_model).pack(pady=10)
                return  # 等待用户选择后再继续
            elif model_type == 'regularization':
                # 创建选择正则化方法的对话框
                reg_window = tk.Toplevel(self.root)
                reg_window.title("选择正则化方法")
                reg_window.geometry("300x250")
                reg_window.resizable(False, False)
                reg_window.grab_set()  # 模态对话框
                
                ttk.Label(reg_window, text="请选择正则化方法:\n", font=("SimHei", 10)).pack(pady=10)
                
                reg_var = tk.StringVar(value="l2")
                
                ttk.Radiobutton(reg_window, text="无正则化", variable=reg_var, value="none").pack(anchor=tk.W, padx=20)
                ttk.Radiobutton(reg_window, text="L1正则化", variable=reg_var, value="l1").pack(anchor=tk.W, padx=20)
                ttk.Radiobutton(reg_window, text="L2正则化", variable=reg_var, value="l2").pack(anchor=tk.W, padx=20)
                
                ttk.Label(reg_window, text="正则化参数 (lambda):").pack(pady=5)
                lambda_entry = ttk.Entry(reg_window, width=10)
                lambda_entry.insert(0, "0.001")
                lambda_entry.pack()
                
                def create_reg_model():
                    regularization = reg_var.get()
                    try:
                        lambda_reg = float(lambda_entry.get())
                    except ValueError:
                        messagebox.showerror("错误", "请输入有效的正则化参数！")
                        return
                    
                    self.model = NeuralNetworkWithRegularization(
                        self.input_size, self.hidden_sizes, self.output_size, 
                        learning_rate=self.learning_rate, 
                        regularization=regularization, 
                        lambda_reg=lambda_reg
                    )
                    reg_window.destroy()
                    self.log_message(f"带{regularization}正则化的神经网络初始化完成！")
                    messagebox.showinfo("成功", f"带{regularization}正则化的神经网络初始化完成！")
                
                ttk.Button(reg_window, text="确定", command=create_reg_model).pack(pady=10)
                return  # 等待用户选择后再继续
            
            self.log_message("神经网络初始化完成！")
            messagebox.showinfo("成功", "神经网络初始化完成！")
        except Exception as e:
            self.log_message(f"创建模型时出错: {e}")
            messagebox.showerror("错误", f"创建模型时出错: {e}")
            
    def train_model(self):
        """训练神经网络模型"""
        if self.model is None:
            messagebox.showwarning("警告", "请先创建或加载模型！")
            return
        
        if self.X_train is None:
            self.load_data()
            if self.X_train is None:  # 如果加载数据失败
                return
        
        # 创建训练参数设置对话框
        train_window = tk.Toplevel(self.root)
        train_window.title("设置训练参数")
        train_window.geometry("300x250")
        train_window.resizable(False, False)
        train_window.grab_set()  # 模态对话框
        
        ttk.Label(train_window, text="设置训练参数:\n", font=("SimHei", 10)).pack(pady=10)
        
        ttk.Label(train_window, text="训练轮数 (epochs):").pack(anchor=tk.W, padx=20)
        epochs_entry = ttk.Entry(train_window, width=10)
        epochs_entry.insert(0, str(self.epochs))
        epochs_entry.pack(anchor=tk.W, padx=20, pady=5)
        
        ttk.Label(train_window, text="批量大小 (batch_size):").pack(anchor=tk.W, padx=20)
        batch_entry = ttk.Entry(train_window, width=10)
        batch_entry.insert(0, str(self.batch_size))
        batch_entry.pack(anchor=tk.W, padx=20, pady=5)
        
        ttk.Label(train_window, text="学习率 (learning_rate):").pack(anchor=tk.W, padx=20)
        lr_entry = ttk.Entry(train_window, width=10)
        lr_entry.insert(0, str(self.learning_rate))
        lr_entry.pack(anchor=tk.W, padx=20, pady=5)
        
        def start_training():
            try:
                self.epochs = int(epochs_entry.get())
                self.batch_size = int(batch_entry.get())
                self.learning_rate = float(lr_entry.get())
                # 更新模型的学习率
                self.model.learning_rate = self.learning_rate
            except ValueError:
                messagebox.showerror("错误", "请输入有效的训练参数！")
                return
            
            train_window.destroy()
            
            self.log_message(f"正在训练神经网络，共{self.epochs}个epoch，批量大小{self.batch_size}，学习率{self.learning_rate}...")
            
            try:
                # 训练模型
                if isinstance(self.model, NeuralNetworkWithExtensions):
                    self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies = \
                        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val, self.epochs, self.batch_size, activation='relu')
                else:
                    self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies = \
                        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val, self.epochs, self.batch_size)
                
                # 评估模型
                self.log_message("正在测试集上评估模型...")
                if isinstance(self.model, NeuralNetworkWithExtensions):
                    test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, activation='relu')
                else:
                    test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test)
                
                result_message = f'测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}'
                self.log_message(result_message)
                
                # 显示训练结果
                self.show_training_results()
                
                self.log_message("模型训练完成！")
                messagebox.showinfo("成功", f"模型训练完成！\n{result_message}")
            except Exception as e:
                self.log_message(f"训练模型时出错: {e}")
                messagebox.showerror("错误", f"训练模型时出错: {e}")
        
        ttk.Button(train_window, text="开始训练", command=start_training).pack(pady=10)
        
    def save_model(self):
        """保存训练好的模型参数"""
        if self.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型！")
            return
        
        # 打开文件对话框选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="保存模型参数",
            defaultextension=".pkl",
            filetypes=[("Pickle文件", "*.pkl")]
        )
        
        if file_path:
            self.log_message(f"保存模型参数到: {file_path}")
            
            try:
                save_model_parameters(self.model, file_path)
                self.log_message("模型参数保存成功！")
                messagebox.showinfo("成功", "模型参数保存成功！")
            except Exception as e:
                self.log_message(f"保存模型参数时出错: {e}")
                messagebox.showerror("错误", f"保存模型参数时出错: {e}")
                
    def load_model(self):
        """加载训练好的模型参数"""
        # 创建或加载模型
        self.log_message("创建模型实例...")
        
        # 创建默认神经网络模型
        self.model = NeuralNetworkWithRegularization(
            self.input_size, self.hidden_sizes, self.output_size,
            learning_rate=self.learning_rate, regularization='l2', lambda_reg=0.001
        )
        
        # 打开文件对话框选择模型文件
        file_path = filedialog.askopenfilename(
            title="加载模型参数",
            filetypes=[("Pickle文件", "*.pkl")]
        )
        
        if file_path:
            self.log_message(f"从{file_path}加载模型参数...")
            
            try:
                if load_trained_model_parameters(self.model, file_path):
                    self.log_message("模型参数加载成功！")
                    messagebox.showinfo("成功", "模型参数加载成功！")
                else:
                    self.log_message("模型参数加载失败！")
                    messagebox.showerror("错误", "模型参数加载失败！")
            except Exception as e:
                self.log_message(f"加载模型参数时出错: {e}")
                messagebox.showerror("错误", f"加载模型参数时出错: {e}")
                
    def recognize_digit(self):
        """识别用户提供的数字图像"""
        # 创建或加载模型
        if self.model is None:
            self.log_message("创建预测模型...")
            
            # 创建神经网络模型
            self.model = NeuralNetworkWithRegularization(
                self.input_size, self.hidden_sizes, self.output_size,
                learning_rate=0.01, regularization='l2', lambda_reg=0.001
            )
            
            # 尝试加载训练好的模型参数
            if os.path.exists('model_weights.pkl'):
                if not load_trained_model_parameters(self.model, 'model_weights.pkl'):
                    messagebox.showwarning("警告", "未找到保存的模型参数，将使用随机初始化的模型进行预测！")
            else:
                messagebox.showwarning("警告", "未找到保存的模型参数，将使用随机初始化的模型进行预测！")
        
        # 打开文件对话框选择图像
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        
        if file_path:
            self.log_message(f"预测图像: {file_path}")
            
            # 清除之前的图形
            for widget in self.display_frame.winfo_children():
                widget.destroy()
            
            try:
                # 预处理图像
                img_flatten, img_array = preprocess_custom_image(file_path)
                if img_flatten is None:
                    messagebox.showerror("错误", "处理图像时出错！")
                    return
                
                # 进行预测
                if hasattr(self.model, 'forward_propagation'):
                    if isinstance(self.model, NeuralNetworkWithExtensions):
                        prediction = self.model.forward_propagation(img_flatten, activation='relu')
                    else:
                        prediction = self.model.forward_propagation(img_flatten)
                    
                    predicted_digit = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # 显示结果
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(img_array, cmap='gray')
                    ax.set_title(f'预测数字: {predicted_digit} (置信度: {confidence:.2f})')
                    ax.axis('off')
                    plt.tight_layout()
                    
                    # 将图表添加到Tkinter窗口
                    canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    result_message = f"预测结果: 数字 {predicted_digit}, 置信度: {confidence:.4f}"
                    self.log_message(result_message)
                    messagebox.showinfo("预测结果", result_message)
                else:
                    self.log_message("模型没有forward_propagation方法")
                    messagebox.showwarning("警告", "模型没有forward_propagation方法")
            except Exception as e:
                self.log_message(f"预测图像时出错: {e}")
                messagebox.showerror("错误", f"预测图像时出错: {e}")
                
    def evaluate_model(self):
        """评估模型性能"""
        if self.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型！")
            return
        
        if self.X_test is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return
        
        self.log_message("在测试集上评估模型...")
        
        try:
            # 检查模型类型，使用适当的评估方法
            if hasattr(self.model, 'evaluate'):
                if isinstance(self.model, NeuralNetworkWithExtensions):
                    test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, activation='relu')
                else:
                    test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test)
                
                result_message = f"测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}"
                self.log_message(result_message)
                messagebox.showinfo("评估结果", result_message)
            else:
                self.log_message("模型没有evaluate方法")
                messagebox.showwarning("警告", "模型没有evaluate方法")
        except Exception as e:
            self.log_message(f"评估模型时出错: {e}")
            messagebox.showerror("错误", f"评估模型时出错: {e}")
            
    def show_training_results(self):
        """显示训练结果"""
        if self.train_losses is None or self.val_losses is None:
            messagebox.showwarning("警告", "没有可用的训练结果！")
            return
        
        # 清除之前的图形
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        try:
            # 创建figure和axes
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle('训练结果')
            
            # 绘制损失曲线
            ax1.plot(self.train_losses, label='训练损失')
            ax1.plot(self.val_losses, label='验证损失')
            ax1.set_title('损失 vs. 轮次')
            ax1.set_xlabel('轮次')
            ax1.set_ylabel('损失')
            ax1.legend()
            
            # 绘制准确率曲线
            ax2.plot(self.train_accuracies, label='训练准确率')
            ax2.plot(self.val_accuracies, label='验证准确率')
            ax2.set_title('准确率 vs. 轮次')
            ax2.set_xlabel('轮次')
            ax2.set_ylabel('准确率')
            ax2.legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 将图表添加到Tkinter窗口
            canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            self.log_message(f"显示训练结果时出错: {e}")
            messagebox.showerror("错误", f"显示训练结果时出错: {e}")
            
    def show_confusion_matrix(self):
        """显示混淆矩阵"""
        if self.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型！")
            return
        
        if self.X_test is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return
        
        self.log_message("生成混淆矩阵...")
        
        # 清除之前的图形
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        try:
            # 获取预测结果
            if hasattr(self.model, 'forward_propagation'):
                if isinstance(self.model, NeuralNetworkWithExtensions):
                    y_pred = self.model.forward_propagation(self.X_test, activation='relu')
                else:
                    y_pred = self.model.forward_propagation(self.X_test)
                
                # 创建figure
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 绘制混淆矩阵
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                
                # 将one-hot编码转换为标签
                y_true_labels = np.argmax(self.y_test, axis=1)
                y_pred_labels = np.argmax(y_pred, axis=1)
                
                # 计算混淆矩阵
                cm = confusion_matrix(y_true_labels, y_pred_labels)
                
                # 绘制混淆矩阵
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=[str(i) for i in range(10)], 
                           yticklabels=[str(i) for i in range(10)],
                           ax=ax)
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title('混淆矩阵')
                
                plt.tight_layout()
                
                # 将图表添加到Tkinter窗口
                canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                self.log_message("混淆矩阵生成完成！")
            else:
                self.log_message("模型没有forward_propagation方法")
                messagebox.showwarning("警告", "模型没有forward_propagation方法")
        except Exception as e:
            self.log_message(f"生成混淆矩阵时出错: {e}")
            messagebox.showerror("错误", f"生成混淆矩阵时出错: {e}")
            
    def visualize_network_weights(self):
        """可视化网络权重"""
        if self.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型！")
            return
        
        # 清除之前的图形
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        try:
            # 检查模型是否有weights属性
            if hasattr(self.model, 'weights'):
                # 创建figure
                fig, axs = plt.subplots(2, 5, figsize=(12, 6))
                fig.suptitle('神经网络权重可视化 (输入层)')
                
                # 选择前10个权重进行可视化
                weights = self.model.weights[0]  # 输入层的权重
                
                if weights.shape[0] == 784:  # 确保是MNIST输入层
                    for i, ax in enumerate(axs.flat):
                        if i < weights.shape[1]:
                            # 将权重重塑为28x28的图像
                            weight_image = weights[:, i].reshape(28, 28)
                            ax.imshow(weight_image, cmap='viridis')
                            ax.set_title(f'神经元 {i+1}')
                        ax.axis('off')
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    
                    # 将图表添加到Tkinter窗口
                    canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                else:
                    self.log_message("仅支持可视化MNIST输入层的权重")
                    messagebox.showwarning("警告", "仅支持可视化MNIST输入层的权重")
            else:
                self.log_message("模型没有weights属性")
                messagebox.showwarning("警告", "模型没有weights属性")
        except Exception as e:
            self.log_message(f"可视化权重时出错: {e}")
            messagebox.showerror("错误", f"可视化权重时出错: {e}")
            
    def show_misclassified_samples(self):
        """显示错误分类的样本"""
        if self.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型！")
            return
        
        if self.X_test is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return
        
        # 清除之前的图形
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        try:
            # 调用main.py中的函数来绘制错误分类的样本
            plot_misclassified_samples(self.model, self.X_test, self.y_test, 25)
            
            self.log_message("错误分类样本已显示！")
        except Exception as e:
            self.log_message(f"显示错误分类样本时出错: {e}")
            messagebox.showerror("错误", f"显示错误分类样本时出错: {e}")
            
    def show_correctly_classified_samples(self):
        """显示正确分类的样本"""
        if self.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型！")
            return
        
        if self.X_test is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return
        
        # 清除之前的图形
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        try:
            # 调用main.py中的函数来绘制正确分类的样本
            plot_correctly_classified_samples(self.model, self.X_test, self.y_test, 25)
            
            self.log_message("正确分类样本已显示！")
        except Exception as e:
            self.log_message(f"显示正确分类样本时出错: {e}")
            messagebox.showerror("错误", f"显示正确分类样本时出错: {e}")
            
    def demo_early_stopping_and_lr_scheduler(self):
        """演示早停策略和学习率调度器"""
        if self.X_train is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return
        
        self.log_message("演示早停策略和学习率调度器...")
        
        # 定义超参数
        input_size = 784
        hidden_sizes = [128, 64]
        output_size = 10
        learning_rate = 0.01
        epochs = 30
        batch_size = 64
        
        # 创建神经网络实例
        self.model = NeuralNetworkWithExtensions(input_size, hidden_sizes, output_size, learning_rate)
        
        # 创建早停策略和学习率调度器
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        lr_scheduler = LearningRateScheduler(initial_lr=learning_rate, decay_type='exponential', decay_rate=0.95)
        
        # 训练模型
        try:
            self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies, learning_rates = \
                self.model.train(
                    self.X_train, self.y_train, self.X_val, self.y_val, epochs, batch_size,
                    activation='leaky_relu', early_stopping=early_stopping, lr_scheduler=lr_scheduler
                )
            
            self.log_message("早停策略和学习率调度器演示完成！")
            
            # 显示训练结果
            self.show_training_results()
            
            # 显示学习率曲线
            self.show_learning_rate_curve(learning_rates)
            
            messagebox.showinfo("成功", "早停策略和学习率调度器演示完成！")
        except Exception as e:
            self.log_message(f"演示时出错: {e}")
            messagebox.showerror("错误", f"演示时出错: {e}")
            
    def compare_activation_functions(self):
        """比较不同激活函数的性能"""
        if self.X_train is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return
        
        self.log_message("比较不同激活函数的性能...")
        
        # 定义超参数
        input_size = 784
        hidden_sizes = [128, 64]
        output_size = 10
        learning_rate = 0.01
        epochs = 10  # 为了快速比较，使用较少的epochs
        batch_size = 64
        
        # 要测试的激活函数
        activation_functions = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu']
        
        results = {}
        
        # 清除之前的图形
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        # 创建figure和axes
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('不同激活函数的性能比较')
        
        try:
            for i, activation in enumerate(activation_functions):
                self.log_message(f"使用 {activation} 激活函数训练模型...")
                
                # 创建神经网络实例
                model = NeuralNetworkWithExtensions(input_size, hidden_sizes, output_size, learning_rate)
                
                # 训练模型
                train_losses, val_losses, train_accuracies, val_accuracies = \
                    model.train(
                        self.X_train, self.y_train, self.X_val, self.y_val, epochs, batch_size,
                        activation=activation
                    )
                
                # 评估模型
                test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, activation=activation)
                self.log_message(f'{activation} - 测试准确率: {test_accuracy:.4f}, 测试损失: {test_loss:.4f}')
                
                # 保存结果
                results[activation] = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy
                }
                
                # 绘制结果
                axs[0, 0].plot(train_losses, label=activation)
                axs[0, 1].plot(val_losses, label=activation)
                axs[1, 0].plot(train_accuracies, label=activation)
                axs[1, 1].plot(val_accuracies, label=activation)
                
                # 更新进度条
                self.update_progress((i + 1) / len(activation_functions) * 100)
            
            # 设置图表属性
            axs[0, 0].set_title('训练损失')
            axs[0, 0].set_xlabel('轮次')
            axs[0, 0].set_ylabel('损失')
            axs[0, 0].legend()
            
            axs[0, 1].set_title('验证损失')
            axs[0, 1].set_xlabel('轮次')
            axs[0, 1].set_ylabel('损失')
            axs[0, 1].legend()
            
            axs[1, 0].set_title('训练准确率')
            axs[1, 0].set_xlabel('轮次')
            axs[1, 0].set_ylabel('准确率')
            axs[1, 0].legend()
            
            axs[1, 1].set_title('验证准确率')
            axs[1, 1].set_xlabel('轮次')
            axs[1, 1].set_ylabel('准确率')
            axs[1, 1].legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 将图表添加到Tkinter窗口
            canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.log_message("不同激活函数性能比较完成！")
            messagebox.showinfo("成功", "不同激活函数性能比较完成！")
        except Exception as e:
            self.log_message(f"比较激活函数时出错: {e}")
            messagebox.showerror("错误", f"比较激活函数时出错: {e}")
            
    def show_learning_rate_curve(self, learning_rates):
        """显示学习率曲线"""
        # 创建一个新的窗口显示学习率曲线
        lr_window = tk.Toplevel(self.root)
        lr_window.title("学习率曲线")
        lr_window.geometry("600x400")
        lr_window.resizable(True, True)
        
        try:
            # 创建figure和axes
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(learning_rates)
            ax.set_title('学习率调度')
            ax.set_xlabel('训练轮次')
            ax.set_ylabel('学习率')
            
            plt.tight_layout()
            
            # 将图表添加到Tkinter窗口
            canvas = FigureCanvasTkAgg(fig, master=lr_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            self.log_message(f"显示学习率曲线时出错: {e}")
            messagebox.showerror("错误", f"显示学习率曲线时出错: {e}")

    def run_parameter_comparison(self):
        """运行参数比较分析工具"""
        if self.X_train is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return
        
        self.log_message("启动参数比较分析工具...")
        
        try:
            # 导入ExperimentManager类
            from demo_comparison_analysis import ExperimentManager
            
            # 创建选择比较类型的对话框
            comparison_window = tk.Toplevel(self.root)
            comparison_window.title("参数比较分析")
            comparison_window.geometry("400x300")
            comparison_window.resizable(False, False)
            comparison_window.grab_set()  # 模态对话框
            
            ttk.Label(comparison_window, text="请选择要运行的比较实验:", font=("SimHei", 10)).pack(pady=15)
            
            comparison_var = tk.StringVar(value="1")
            
            ttk.Radiobutton(comparison_window, text="1. 比较不同优化算法 (SGD, Momentum, Adam)", variable=comparison_var, value="1").pack(anchor=tk.W, padx=20, pady=5)
            ttk.Radiobutton(comparison_window, text="2. 比较不同损失函数 (交叉熵, 均方误差)", variable=comparison_var, value="2").pack(anchor=tk.W, padx=20, pady=5)
            ttk.Radiobutton(comparison_window, text="3. 比较不同正则化方法 (无正则化, L1, L2)", variable=comparison_var, value="3").pack(anchor=tk.W, padx=20, pady=5)
            ttk.Radiobutton(comparison_window, text="4. 运行所有比较实验", variable=comparison_var, value="4").pack(anchor=tk.W, padx=20, pady=5)
            
            def start_comparison():
                choice = comparison_var.get()
                comparison_window.destroy()
                
                try:
                    # 创建实验管理器
                    manager = ExperimentManager()
                    
                    # 根据选择运行相应的比较实验
                    if choice == '1':
                        self.log_message("开始比较不同优化算法的效果...")
                        manager.compare_optimizers()
                    elif choice == '2':
                        self.log_message("开始比较不同损失函数的效果...")
                        manager.compare_loss_functions()
                    elif choice == '3':
                        self.log_message("开始比较不同正则化方法的效果...")
                        manager.compare_regularization()
                    elif choice == '4':
                        self.log_message("开始运行所有比较实验...")
                        manager.run_all_comparisons()
                    
                    self.log_message("参数比较分析完成！")
                    messagebox.showinfo("成功", "参数比较分析完成！\n结果图表已保存到当前目录。")
                except Exception as e:
                    self.log_message(f"参数比较分析时出错: {e}")
                    messagebox.showerror("错误", f"参数比较分析时出错: {e}")
            
            ttk.Button(comparison_window, text="开始分析", command=start_comparison).pack(pady=15)
            
        except ImportError:
            self.log_message("无法导入demo_comparison_analysis模块！")
            messagebox.showerror("错误", "无法导入demo_comparison_analysis模块！")
        except Exception as e:
            self.log_message(f"启动参数比较分析工具时出错: {e}")
            messagebox.showerror("错误", f"启动参数比较分析工具时出错: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedNeuralNetworkApp(root)
    root.mainloop()