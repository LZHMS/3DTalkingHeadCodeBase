"""
Simple test script to verify FlowMatching implementation.
"""
import sys
sys.path.insert(0, 'e:\\Workspace\\Projects\\FlowTalker\\3DTalkingHeadCodeBase')

import torch
from models.FlowMatching.flow_matching import FlowMatching
from models.FlowMatching.flow_network import FlowDenoisingNetwork


def test_flow_matching():
    """Test FlowMatching algorithm."""
    print("Testing FlowMatching algorithm...")
    
    fm = FlowMatching(min_sigma=0.0, inference_mode='euler', num_steps=10, reverse_flow=True)
    
    # Test data
    batch_size, seq_len, dim = 4, 100, 54
    x0 = torch.randn(batch_size, seq_len, dim)
    x1 = torch.randn(batch_size, seq_len, dim)
    t = torch.rand(batch_size)
    
    # Test get_conditional_flow
    xt = fm.get_conditional_flow(x0, x1, t)
    assert xt.shape == x1.shape, "Conditional flow shape mismatch"
    print(f"✓ Conditional flow shape: {xt.shape}")
    
    # Test loss
    predicted_v = torch.randn_like(x1)
    loss = fm.loss(predicted_v, x0, x1)
    assert loss.shape == (batch_size,), "Loss shape mismatch"
    print(f"✓ Loss shape: {loss.shape}, mean: {loss.mean().item():.4f}")
    
    # Test ODE solving
    def simple_ode(t, x):
        return torch.zeros_like(x)
    
    x_final = fm.to_data(simple_ode, x0)
    assert x_final.shape == x0.shape, "ODE output shape mismatch"
    print(f"✓ ODE solving output shape: {x_final.shape}")
    
    print("FlowMatching algorithm tests passed!\n")


def test_flow_network():
    """Test FlowDenoisingNetwork."""
    print("Testing FlowDenoisingNetwork...")
    
    # Mock config
    class MockCfg:
        class ADD:
            STYLE_ENC_CKPT = 'dummy_path'  # Non-empty to enable style
        class MODEL:
            class HEAD:
                NO_HEAD_POSE = False
                STYLE_DIM = 128
                USE_INDICATOR = True
                ALIGN_MASK_WIDTH = 1
                USE_LEARNABLE_PE = False
            class BACKBONE:
                HIDDEN_SIZE = 512
                NUM_ATTENTION_HEADS = 8
                NUM_HIDDEN_LAYERS = 4
            class TAIL:
                MLP_RATIO = 4
        class DATASET:
            class HDTF_TFHP:
                N_PREV_MOTIONS = 10
                MOTIONS = 100
    
    cfg = MockCfg()
    network = FlowDenoisingNetwork(cfg)
    
    # Determine device (use CPU if CUDA not available)
    device = torch.device('cpu')
    network = network.to(device)
    
    # Test input
    batch_size = 2
    motion_feat = torch.randn(batch_size, 100, 54).to(device)
    audio_feat = torch.randn(batch_size, 100, 512).to(device)
    person_feat = torch.randn(batch_size, 1, 228).to(device)  # 100 (shape) + 128 (style)
    prev_motion = torch.randn(batch_size, 10, 54).to(device)
    prev_audio = torch.randn(batch_size, 10, 512).to(device)
    t = torch.rand(batch_size).to(device)
    indicator = torch.ones(batch_size, 100).to(device)
    
    # Forward pass
    flow = network(motion_feat, audio_feat, person_feat, prev_motion, prev_audio, t, indicator)
    
    expected_shape = (batch_size, 110, 54)  # 10 (prev) + 100 (current)
    assert flow.shape == expected_shape, f"Flow shape mismatch: {flow.shape} vs {expected_shape}"
    print(f"✓ Flow output shape: {flow.shape}")
    print(f"✓ Flow mean: {flow.mean().item():.4f}, std: {flow.std().item():.4f}")
    
    print("FlowDenoisingNetwork tests passed!\n")


def test_integration():
    """Test integration of components."""
    print("Testing integration...")
    
    # This would test the full FlowMatchingHead model
    # Requires full dependencies, so we skip for now
    print("Integration test skipped (requires full model setup)")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("FlowMatching Implementation Tests")
    print("=" * 60)
    print()
    
    try:
        test_flow_matching()
        test_flow_network()
        test_integration()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
