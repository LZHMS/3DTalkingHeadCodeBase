"""
FlowMatching 使用示例
演示如何使用迁移后的 FlowMatching 模型进行训练和推理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from pathlib import Path

# 示例 1: 基础训练循环
def example_training():
    """演示 FlowMatching 的基础训练流程"""
    print("=" * 60)
    print("示例 1: FlowMatching 训练流程")
    print("=" * 60)
    
    # 模拟配置
    class MockConfig:
        class ADD:
            STYLE_ENC_CKPT = 'dummy_path'  # Non-empty to enable style
        class MODEL:
            class HEAD:
                ROT_REPR = 'aa'
                NO_HEAD_POSE = False
                AUDIO_MODEL = 'wav2vec2'
                AUDIO_DIM = 128
                STYLE_DIM = 128
                USE_INDICATOR = True
                ALIGN_MASK_WIDTH = 1
                USE_LEARNABLE_PE = False
            class BACKBONE:
                HIDDEN_SIZE = 512
                NUM_ATTENTION_HEADS = 8
                NUM_HIDDEN_LAYERS = 4
                MIN_SIGMA = 0.0
                INFERENCE_MODE = 'euler'
                NUM_STEPS = 10
                REVERSE_FLOW = True
                LOG_NORMAL_MEAN = 0.0
                LOG_NORMAL_STD = 1.0
            class TAIL:
                MLP_RATIO = 4
        class DATASET:
            class HDTF_TFHP:
                COEF_FPS = 25
                MOTIONS = 100
                N_PREV_MOTIONS = 10
                USE_CONTEXT_AUDIO = True
    
    cfg = MockConfig()
    
    # 由于依赖项限制,这里只展示代码结构
    print("\n训练代码结构:")
    print("""
    from models import FlowMatchingHead
    from trainers import FlowMatchingTrainer
    
    # 1. 初始化模型
    model = FlowMatchingHead(cfg)
    model = model.to('cuda')
    
    # 2. 准备数据
    for batch in train_loader:
        audio = batch['audio']
        motion = batch['motion']
        shape = batch['shape']
        style = batch['style']
        
        # 3. 前向传播
        predicted_v, target_v, _, _ = model(
            motion, audio, shape, style
        )
        
        # 4. 计算损失
        loss = F.mse_loss(predicted_v, target_v)
        
        # 5. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """)
    
    print("\n✓ 训练流程示例完成")


# 示例 2: Flow Matching 核心算法
def example_flow_matching():
    """演示 Flow Matching 核心算法"""
    print("\n" + "=" * 60)
    print("示例 2: Flow Matching 核心算法")
    print("=" * 60)
    
    from models.FlowMatching.flow_matching import FlowMatching
    
    # 初始化 Flow Matching
    fm = FlowMatching(
        min_sigma=0.0,
        inference_mode='euler',
        num_steps=10,
        reverse_flow=True
    )
    
    print("\n1. 生成测试数据:")
    batch_size, seq_len, dim = 2, 100, 54
    x0 = torch.randn(batch_size, seq_len, dim)  # 噪声
    x1 = torch.randn(batch_size, seq_len, dim)  # 数据
    t = torch.rand(batch_size)                   # 时间
    
    print(f"   x0 (噪声): {x0.shape}")
    print(f"   x1 (数据): {x1.shape}")
    print(f"   t (时间): {t.shape}, 范围: [{t.min():.2f}, {t.max():.2f}]")
    
    print("\n2. 计算条件流 (插值):")
    xt = fm.get_conditional_flow(x0, x1, t)
    print(f"   x_t = (1-t)·x1 + t·x0")
    print(f"   x_t: {xt.shape}")
    
    print("\n3. 计算目标速度:")
    target_v = x0 - x1  # 反向流
    print(f"   v_target = x0 - x1")
    print(f"   v_target: {target_v.shape}")
    
    print("\n4. 模拟网络预测:")
    predicted_v = torch.randn_like(target_v)
    print(f"   v_predicted: {predicted_v.shape}")
    
    print("\n5. 计算损失:")
    loss = fm.loss(predicted_v, x0, x1)
    print(f"   loss = MSE(v_predicted, v_target)")
    print(f"   loss: {loss.shape}, 平均值: {loss.mean():.4f}")
    
    print("\n6. ODE 采样 (简化):")
    def dummy_ode(t, x):
        return torch.zeros_like(x)
    
    x_sampled = fm.to_data(dummy_ode, x0)
    print(f"   从 x0 采样到 x1")
    print(f"   x_sampled: {x_sampled.shape}")
    
    print("\n✓ Flow Matching 算法演示完成")


# 示例 3: 对比 Diffusion 和 Flow Matching
def example_comparison():
    """对比 Diffusion 和 Flow Matching"""
    print("\n" + "=" * 60)
    print("示例 3: Diffusion vs Flow Matching 对比")
    print("=" * 60)
    
    print("\n时间参数化对比:")
    print("-" * 60)
    
    # Diffusion 时间
    print("\nDiffusion (离散时间):")
    num_steps = 1000
    t_diffusion = torch.randint(0, num_steps, (8,))
    print(f"   采样: {t_diffusion.tolist()}")
    print(f"   范围: [0, {num_steps}]")
    
    # Flow Matching 时间
    print("\nFlow Matching (连续时间):")
    t_flow = torch.rand(8)
    print(f"   采样: {['%.3f' % t for t in t_flow.tolist()]}")
    print(f"   范围: [0.0, 1.0]")
    
    print("\n前向过程对比:")
    print("-" * 60)
    
    batch_size = 4
    x0 = torch.randn(batch_size, 10)
    x1 = torch.randn(batch_size, 10)
    
    # Diffusion
    print("\nDiffusion:")
    print("   x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε")
    alpha_bar = 0.5
    noise = torch.randn_like(x0)
    xt_diffusion = torch.sqrt(torch.tensor(alpha_bar)) * x1 + \
                   torch.sqrt(torch.tensor(1 - alpha_bar)) * noise
    print(f"   输出: {xt_diffusion.shape}")
    
    # Flow Matching
    print("\nFlow Matching:")
    print("   x_t = (1-t) · x1 + t · x0")
    t = torch.rand(batch_size)
    xt_flow = (1 - t.view(-1, 1)) * x1 + t.view(-1, 1) * x0
    print(f"   输出: {xt_flow.shape}")
    
    print("\n训练目标对比:")
    print("-" * 60)
    
    print("\nDiffusion:")
    print("   预测: 噪声 ε 或样本 x0")
    print("   损失: ||ε_pred - ε|| 或 ||x0_pred - x0||")
    
    print("\nFlow Matching:")
    print("   预测: 速度场 v_t")
    print("   损失: ||v_pred - (x0 - x1)||")
    
    print("\n采样步数对比:")
    print("-" * 60)
    print("""
    方法              | 步数  | 相对速度
    ----------------+-------+---------
    DDPM            | 1000  | 最慢
    DDIM            | 50    | 中等
    Flow (Euler)    | 25    | 快
    Flow (Adaptive) | 自适应 | 中等
    """)
    
    print("✓ 对比演示完成")


# 示例 4: 实际使用场景
def example_use_case():
    """实际使用场景示例"""
    print("\n" + "=" * 60)
    print("示例 4: 实际使用场景")
    print("=" * 60)
    
    print("\n场景 1: 快速原型验证")
    print("-" * 60)
    print("""
    配置文件修改:
    MODEL:
      BACKBONE:
        NUM_STEPS: 10  # 快速采样
    
    TRAIN:
      MAX_ITERS: 10000  # 快速训练
    
    优点: 快速迭代,验证想法
    """)
    
    print("\n场景 2: 高质量生成")
    print("-" * 60)
    print("""
    配置文件修改:
    MODEL:
      BACKBONE:
        NUM_STEPS: 50          # 更多采样步骤
        INFERENCE_MODE: euler  # 确定性采样
    
    OPTIM:
      LR: 0.00005  # 更稳定的训练
    
    优点: 更好的生成质量
    """)
    
    print("\n场景 3: 实时应用")
    print("-" * 60)
    print("""
    配置文件修改:
    MODEL:
      BACKBONE:
        NUM_STEPS: 5           # 最少步骤
        INFERENCE_MODE: euler  # 最快求解
    
    部署优化:
    - 使用 torch.compile()
    - 使用 TensorRT
    - 批量推理
    
    优点: 低延迟,适合实时应用
    """)
    
    print("✓ 使用场景演示完成")


# 主函数
def main():
    """运行所有示例"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "FlowMatching 使用示例" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # 运行示例
        example_training()
        example_flow_matching()
        example_comparison()
        example_use_case()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成! ✓")
        print("=" * 60)
        
        print("\n下一步:")
        print("1. 查看 QUICKSTART.md 了解如何开始训练")
        print("2. 查看 COMPARISON.md 了解详细对比")
        print("3. 查看 models/FlowMatching/README.md 了解技术细节")
        print("4. 运行 test_flowmatching.py 进行单元测试")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
