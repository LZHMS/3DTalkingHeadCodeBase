# FlowMatching 快速开始指南

## 环境准备

### 1. 安装依赖
```bash
# 基础依赖 (如果还未安装)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Flow Matching 特定依赖
pip install torchdiffeq

# 其他依赖
pip install transformers
pip install tensorboard
pip install wandb  # 可选,用于实验跟踪
```

### 2. 数据准备
确保数据目录结构如下:
```
data/
├── HDTF_TFHP/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── stats_train.npz
│   └── coefficients/
│       ├── person1/
│       └── person2/
│       └── ...
└── audio/
    └── ...
```

## 训练 FlowMatching 模型

### Step 1: 训练风格编码器 (如果还未训练)

```bash
python main/train.py \
  --config-file config/style_trainer_config.yaml \
  --gpu 0 \
  --use-wandb \
  --wandb-name "StyleEncoder_Training"
```

### Step 2: 训练 FlowMatching 模型

```bash
python main/train.py \
  --config-file config/flowmatching_trainer_config.yaml \
  --gpu 0 \
  --use-wandb \
  --wandb-name "FlowMatching_Training"
```

### Step 3: 监控训练

训练过程中可以通过以下方式监控:

1. **TensorBoard**:
```bash
tensorboard --logdir output/
```

2. **WandB** (如果启用):
访问 https://wandb.ai 查看实验

## 配置调整

### 基础配置 (config/flowmatching_trainer_config.yaml)

```yaml
# 修改数据路径
DATASET:
  ROOT: ./your/data/path

# 修改风格编码器路径
ADD:
  STYLE_ENC_CKPT: ./path/to/style_encoder.pth

# 调整批量大小
DATALOADER:
  TRAIN:
    BATCH_SIZE: 32  # 根据 GPU 内存调整

# 调整学习率
OPTIM:
  LR: 0.0001
  WEIGHT_DECAY: 0.01
```

### Flow Matching 特定参数

```yaml
MODEL:
  BACKBONE:
    # ODE 求解器设置
    INFERENCE_MODE: 'euler'  # 或 'adaptive'
    NUM_STEPS: 25            # Euler 步数,越多越慢但质量可能更好
    
    # 流方向
    REVERSE_FLOW: True       # True: x1->x0, False: x0->x1
    
    # 时间采样
    LOG_NORMAL_MEAN: 0.0     # 调整时间分布
    LOG_NORMAL_STD: 1.0      # 标准差越大,时间分布越广
```

## 推理/采样

### 使用训练好的模型生成

```python
import torch
from models import FlowMatchingHead

# 加载模型
model = FlowMatchingHead(cfg).to('cuda')
model.load_state_dict(torch.load('path/to/checkpoint.pth'))
model.eval()

# 准备输入
audio = torch.randn(1, 16000 * 4).to('cuda')  # 4秒音频
shape_feat = torch.randn(1, 100).to('cuda')
style_feat = torch.randn(1, 128).to('cuda')

# 采样
with torch.no_grad():
    motion, _, _ = model.sample(
        audio, 
        shape_feat, 
        style_feat,
        cfg_scale=1.15,  # CFG 强度
        cfg_cond=['audio', 'style']  # 使用的条件
    )

# motion: (1, 100, 54) - 生成的运动系数
```

## 常见问题

### Q1: 训练时 GPU 内存不足

**解决方案**:
1. 减小批量大小
```yaml
DATALOADER:
  TRAIN:
    BATCH_SIZE: 16  # 从 32 减到 16
```

2. 减少序列长度
```yaml
DATASET:
  HDTF_TFHP:
    MOTIONS: 50  # 从 100 减到 50
```

### Q2: 采样速度慢

**解决方案**:
1. 减少 ODE 步数
```yaml
MODEL:
  BACKBONE:
    NUM_STEPS: 10  # 从 25 减到 10
```

2. 使用 Euler 而非 Adaptive
```yaml
MODEL:
  BACKBONE:
    INFERENCE_MODE: 'euler'
```

### Q3: 训练不稳定

**解决方案**:
1. 调整学习率
```yaml
OPTIM:
  LR: 0.00005  # 减小学习率
```

2. 增加 warmup
```yaml
OPTIM:
  LR_SCHEDULER: "gradual"
  WARMUP_MULTIPLIER: 1
```

### Q4: 生成质量不佳

**解决方案**:
1. 调整 CFG 强度
```python
# 在采样时
cfg_scale=1.5  # 增加 CFG 强度 (1.0-2.0)
```

2. 增加采样步数
```yaml
MODEL:
  BACKBONE:
    NUM_STEPS: 50  # 增加步数
```

3. 调整时间采样分布
```yaml
MODEL:
  BACKBONE:
    LOG_NORMAL_MEAN: -1.0  # 偏向早期时间
    LOG_NORMAL_STD: 1.5    # 增加多样性
```

## 性能优化

### 训练加速

1. **使用混合精度训练** (如果支持):
```python
# 在 trainer 中启用 AMP
with torch.cuda.amp.autocast():
    loss = model(...)
```

2. **多 GPU 训练**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main/train.py \
  --config-file config/flowmatching_trainer_config.yaml
```

### 采样加速

1. 使用编译优化 (PyTorch 2.0+):
```python
model = torch.compile(model)
```

2. 批量采样:
```python
# 一次采样多个样本
batch_size = 8
audio_batch = torch.randn(batch_size, 16000 * 4).to('cuda')
motions = model.sample(audio_batch, ...)
```

## 评估

### 定量评估
```bash
python main/train.py \
  --config-file config/flowmatching_trainer_config.yaml \
  --eval-only \
  --model-dir output/checkpoints/iter_200000.pth
```

### 可视化结果
可以使用现有的可视化工具查看生成的运动:
```python
from utils.visualization import visualize_motion

visualize_motion(motion, audio, save_path='result.mp4')
```

## 调试技巧

### 启用调试模式
```bash
python main/train.py \
  --config-file config/flowmatching_trainer_config.yaml \
  --debug
```

### 检查梯度
```python
# 在 trainer 中添加
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'{name}: {param.grad.norm()}')
```

### 可视化时间分布
```python
import matplotlib.pyplot as plt

# 采样时间
t = model._log_normal_sample(10000, 'cuda')
plt.hist(t.cpu().numpy(), bins=50)
plt.savefig('time_distribution.png')
```

## 进阶使用

### 自定义损失函数

在 `flowmatching_trainer.py` 中修改:
```python
def forward_backward(self, batch):
    # ... 现有代码 ...
    
    # 添加自定义损失
    custom_loss = your_custom_loss_function(predicted_v, target_v)
    loss = loss_dict['flow'] + 0.1 * custom_loss
    
    return loss_dict
```

### 修改网络架构

在 `flow_network.py` 中:
```python
# 修改 Transformer 层数
self.n_layers = 12  # 从 8 增加到 12

# 修改隐藏维度
self.feature_dim = 768  # 从 512 增加到 768
```

## 资源链接

- **论文**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **原始实现**: MeanAudio 项目
- **问题反馈**: 请在 GitHub Issues 提交

## 下一步

1. 尝试不同的超参数组合
2. 实验不同的 ODE 求解器
3. 添加自定义条件信息
4. 与 DiffPoseTalk 进行对比实验

祝训练顺利! 🚀
