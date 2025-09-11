# Backpropagation From Scratch

本项目从零开始实现神经网络的反向传播算法，详细展示了梯度计算和参数更新的每一个步骤。这是一个完整的教学项目，用于深入理解神经网络的核心原理。

## 项目结构

- `model.py` - 手动实现的神经网络核心组件
- `train.py` - 训练脚本，展示不同任务的训练过程
- `demo.py` - 详细演示反向传播的每一步计算
- `visualize.py` - 可视化脚本，生成各种图表和动画
- `README.md` - 项目说明文档

## 快速开始

### 1. 安装依赖
```bash
pip install numpy matplotlib scikit-learn networkx
```

### 2. 运行训练演示
```bash
python train.py
```
展示二分类、回归和激活函数对比三个完整示例。

### 3. 查看详细的反向传播过程
```bash
python demo.py
```
逐步展示XOR问题的求解过程，包括每一层的梯度计算。

### 4. 生成可视化图表
```bash
python visualize.py
```
生成网络架构图、损失曲面、激活函数图等多种可视化。

## 核心特性

### 🧠 完整的神经网络实现
- **激活函数**: ReLU, Sigmoid, Tanh (包含前向和反向传播)
- **损失函数**: 均方误差 (MSE), 二元交叉熵 (BCE)
- **全连接层**: 权重初始化、前向传播、梯度计算
- **网络架构**: 灵活的层级结构设计

### 📊 详细的训练分析
- **实时训练监控**: 损失、准确率、梯度幅度跟踪
- **权重更新可视化**: 显示每层权重的变化过程
- **梯度流分析**: 检测梯度消失/爆炸问题
- **学习曲线**: 训练过程的完整可视化

### 🔍 教学友好的演示
- **逐步计算展示**: 显示每一层的前向和反向传播计算
- **数学公式对照**: 代码实现与理论公式的对应关系
- **经典问题求解**: XOR问题的详细解析过程
- **参数敏感性分析**: 不同学习率和架构的影响

## 主要组件详解

### model.py 核心实现

#### 激活函数类
```python
class ReLU(ActivationFunction):
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x):
        return (x > 0).astype(float)
```

#### 全连接层
```python
class Dense(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        # 计算权重梯度: ∂L/∂W = input^T × output_gradient
        weight_gradients = np.dot(self.input.T, output_gradient)
        # 计算偏置梯度: ∂L/∂b = sum(output_gradient)
        bias_gradients = np.sum(output_gradient, axis=0, keepdims=True)
        # 计算输入梯度: ∂L/∂input = output_gradient × W^T
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # 更新参数
        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * bias_gradients
        
        return input_gradient
```

#### 完整网络
```python
class NeuralNetwork:
    def backward(self, y_true, y_pred, learning_rate):
        # 从损失函数开始计算初始梯度
        gradient = self.loss_function.backward(y_true, y_pred)
        
        # 反向传播梯度
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
```

## 演示案例

### 1. 二分类问题 (同心圆数据集)
```python
# 生成非线性可分的数据
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5)

# 创建网络
network = create_simple_classifier(input_size=2, hidden_size=10, output_size=1)

# 训练
history = network.train(X, y, epochs=500, learning_rate=0.1)
```

**学习要点**:
- 非线性激活函数的必要性
- 决策边界的形成过程
- 分类准确率的提升曲线

### 2. 回归问题 (正弦函数拟合)
```python
# 生成正弦函数数据
X = np.linspace(-2*np.pi, 2*np.pi, 500).reshape(-1, 1)
y = np.sin(X) + noise

# 创建回归网络
network = create_regression_network(input_size=1, hidden_sizes=[20, 20], output_size=1)
```

**学习要点**:
- 函数逼近能力
- 过拟合和欠拟合现象
- 网络深度对拟合能力的影响

### 3. XOR问题 (经典演示)
```python
# XOR数据集
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 详细展示每一步计算
network.detailed_forward(X, verbose=True)
network.detailed_backward(y, predictions, learning_rate, verbose=True)
```

**学习要点**:
- 线性不可分问题的解决
- 隐藏层的作用机制
- 梯度如何在网络中传播

## 可视化功能

### 1. 网络架构图
- 神经元和连接的可视化
- 层级结构展示
- 激活函数标注

### 2. 激活函数分析
- 函数曲线和导数曲线
- 梯度消失问题演示
- 不同激活函数的比较

### 3. 损失曲面可视化
- 3D损失曲面
- 等高线图
- 梯度下降路径

### 4. 学习动态分析
- 不同学习率的收敛行为
- 权重更新轨迹
- 训练稳定性分析

### 5. 梯度流可视化
- 各层梯度大小
- 权重更新幅度
- 梯度传播方向

## 数学原理

### 反向传播核心公式

1. **链式法则基础**
   ```
   ∂L/∂w_{ij}^{(l)} = ∂L/∂a_j^{(l)} × ∂a_j^{(l)}/∂z_j^{(l)} × ∂z_j^{(l)}/∂w_{ij}^{(l)}
   ```

2. **梯度递归关系**
   ```
   δ^{(l)} = (W^{(l+1)})^T δ^{(l+1)} ⊙ σ'(z^{(l)})
   ```

3. **参数更新规则**
   ```
   W^{(l)} := W^{(l)} - α × δ^{(l+1)} × (a^{(l)})^T
   b^{(l)} := b^{(l)} - α × δ^{(l+1)}
   ```

### 损失函数梯度

**均方误差 (MSE)**:
```
∂L/∂y_pred = 2(y_pred - y_true) / N
```

**二元交叉熵 (BCE)**:
```
∂L/∂y_pred = -(y_true/y_pred - (1-y_true)/(1-y_pred)) / N
```

## 实验结果

### 性能表现
- **XOR问题**: 100% 准确率 (3层网络，50轮训练)
- **同心圆分类**: ~95% 测试准确率 (隐藏层10个神经元)
- **正弦函数拟合**: MSE < 0.01 (2个隐藏层，各20个神经元)

### 训练效率
- **CPU训练**: 小规模问题秒级收敛
- **内存使用**: 基于NumPy，内存效率高
- **数值稳定性**: 包含梯度裁剪和权重初始化

## 教学价值

### 适合学习者
- 机器学习初学者
- 深度学习课程学生  
- 想要理解反向传播原理的开发者
- 神经网络研究人员

### 学习路径建议
1. **基础概念**: 先运行 `train.py` 了解整体流程
2. **详细分析**: 运行 `demo.py` 查看计算细节
3. **可视化理解**: 运行 `visualize.py` 生成图表
4. **代码研读**: 仔细阅读 `model.py` 的实现
5. **实验拓展**: 尝试修改网络结构和参数

### 扩展实验建议

#### 1. 网络架构实验
```python
# 比较不同隐藏层大小
sizes = [5, 10, 20, 50]
for size in sizes:
    network = create_simple_classifier(2, size, 1)
    # 训练并比较性能
```

#### 2. 学习率调度
```python
# 实现学习率衰减
def train_with_schedule(network, X, y, initial_lr=0.1, decay=0.95):
    lr = initial_lr
    for epoch in range(epochs):
        network.train_step(X, y, lr)
        lr *= decay
```

#### 3. 正则化技术
```python
# 添加L2正则化
def l2_regularization(weights, lambda_reg):
    return lambda_reg * np.sum([np.sum(w**2) for w in weights])
```

## 与现代框架对比

### 本项目优势
- **透明性**: 每一步计算都可见和可控
- **教学性**: 代码结构清晰，便于理解
- **轻量级**: 无复杂依赖，易于修改和实验

### PyTorch对比示例
```python
# 本项目实现
network = create_simple_classifier(2, 10, 1)
predictions = network.forward(X)
network.backward(y, predictions, 0.1)

# PyTorch实现  
import torch.nn as nn
model = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
loss = nn.MSELoss()(model(X_torch), y_torch)
loss.backward()
```

## 常见问题解答

### Q: 为什么不使用现成的深度学习框架?
A: 本项目的目的是教学和理解。通过手动实现，可以深入理解每个组件的工作原理，这对于调试和优化深度学习模型非常重要。

### Q: 这个实现的性能如何?
A: 这个实现专注于可读性和教学价值，而不是性能。对于大规模问题，建议使用PyTorch或TensorFlow。

### Q: 如何扩展到更复杂的架构?
A: 可以添加新的层类型（如卷积层、LSTM层）和正则化技术（如Dropout、BatchNorm）。代码结构设计为易于扩展。

### Q: 数值稳定性如何保证?
A: 实现中包含了梯度裁剪、权重初始化和激活函数输入范围限制等技术来保证数值稳定性。

## 参考资料

### 理论基础
- [Deep Learning (Ian Goodfellow)](http://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/)

### 数学背景
- [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/)
- [The Matrix Cookbook](https://www2.imm.dtu.dk/pubdb/pubs/3274-full.html)

### 实现参考
- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)
- [Neural Networks from Scratch](https://nnfs.io/)

## 贡献指南

欢迎提交改进建议和问题报告！可以改进的方向：
- 添加更多激活函数（Leaky ReLU, Swish等）
- 实现批量归一化和Dropout
- 添加更多优化算法（Adam, RMSprop等）
- 创建更多可视化功能
- 改进数值稳定性

## 许可证

本项目采用MIT许可证。欢迎在教学和研究中自由使用。