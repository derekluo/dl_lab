# Transformer From Scratch

本项目展示了如何从零开始实现基本的 Transformer 模型，用于文本分类任务。

## 项目结构

- `model.py` - Transformer 模型实现，包含所有核心组件
- `train.py` - 训练脚本，生成训练结果图表
- `inference.py` - 推理脚本，用于测试训练好的模型
- `generate_test_texts.py` - 生成测试文本数据的脚本
- `.gitignore` - Git 忽略文件配置
- `README.md` - 项目说明文档

## 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

### 2. 训练模型
```bash
python train.py
```
训练完成后会生成：
- `transformer_model.pth` - 训练好的模型文件
- `training_results.png` - 训练过程可视化图表

### 3. 生成测试数据
```bash
python generate_test_texts.py
```
会在 `test_texts/` 目录下生成各类别的测试文本。

### 4. 测试模型
```bash
python inference.py
```
测试预定义样本并进入交互模式。

## 模型架构

### 核心组件

1. **MultiHeadAttention** - 多头注意力机制
   - 实现缩放点积注意力
   - 支持多头并行计算
   - 包含线性投影层

2. **PositionalEncoding** - 位置编码
   - 使用正弦和余弦函数
   - 为序列提供位置信息

3. **TransformerBlock** - Transformer 块  
   - 多头注意力 + 前馈网络
   - 残差连接和层归一化
   - 支持掩码机制

4. **SimpleTransformer** - 完整模型
   - 词嵌入层和位置编码
   - 多个 Transformer 块
   - 文本分类头

### 默认参数
- 词汇表大小: 5,000
- 模型维度: 128
- 注意力头数: 4  
- Transformer 层数: 2
- 分类类别: 5 (技术、体育、科学、音乐、食物)
- 最大序列长度: 64

## 数据集

训练使用合成的文本数据，包含5个类别：

1. **Technology** - 计算机、编程、AI等相关内容
2. **Sports** - 体育运动、比赛、运动员等
3. **Science** - 科学研究、实验、理论等  
4. **Music** - 音乐、乐器、演出等
5. **Food** - 食物、烹饪、餐厅等

## 文件说明

### model.py
包含完整的 Transformer 实现：
- `MultiHeadAttention`: 多头注意力机制
- `PositionalEncoding`: 位置编码
- `FeedForward`: 前馈网络
- `TransformerBlock`: Transformer 块
- `Transformer`: 完整模型
- `SimpleTransformer`: 简化接口

### train.py  
训练脚本功能：
- 生成合成训练数据
- 构建词汇表
- 训练模型并保存
- 生成训练过程图表
- 支持 CPU/GPU/MPS 加速

### inference.py
推理脚本功能：  
- 加载训练好的模型
- 对新文本进行分类
- 预定义样本测试
- 交互式文本分类

### generate_test_texts.py
测试数据生成器：
- 为每个类别生成丰富的测试文本
- 使用模板和随机组合
- 保存为多种格式（TXT/JSON）
- 创建带标签的混合测试集

## 使用示例

### 训练自定义数据
```python
from model import SimpleTransformer
import torch

# 自定义模型配置
model = SimpleTransformer(
    vocab_size=10000,
    d_model=256,
    n_heads=8,
    n_layers=4,
    num_classes=3,
    max_length=128
)
```

### 单个文本预测
```python
from inference import predict_text, get_class_name

text = "This machine learning algorithm improves performance"
class_id, confidence = predict_text(text)
print(f"预测类别: {get_class_name(class_id)}, 置信度: {confidence:.2%}")
```

## 训练结果

训练脚本会自动生成 `training_results.png`，包含：
- 训练损失曲线
- 测试准确率曲线

典型性能：
- 训练准确率: >95%
- 测试准确率: >90%
- 训练时间: 2-5分钟 (CPU)

## 扩展建议

1. **数据集**
   - 使用真实文本数据集 (IMDB, AG News等)
   - 添加数据增强技术
   - 支持多语言文本

2. **模型优化**  
   - 增加模型层数和维度
   - 实现学习率调度
   - 添加权重衰减和dropout

3. **功能扩展**
   - 序列到序列任务
   - 预训练和微调
   - 注意力可视化

## 注意事项

- 这是一个教学用的简化实现，展示 Transformer 核心原理
- 生产环境建议使用 Hugging Face Transformers 库  
- 模型相对较小，适合在 CPU 上快速演示
- 合成数据主要用于概念验证，实际应用需要真实数据

## 技术细节

### 注意力机制
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

### 位置编码
```
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
```

### 层归一化
在每个子层后应用，有助于训练稳定性。

### 残差连接
```
output = LayerNorm(x + Sublayer(x))
```