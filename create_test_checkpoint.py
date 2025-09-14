#!/usr/bin/env python3
"""
Create a test checkpoint for hierarchical MoE
"""

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def create_checkpoint():
    print("Creating hierarchical MoE test checkpoint...")
    
    # Create config matching eval_model.py defaults
    config = MiniMindConfig(
        vocab_size=6400,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=None,  # Let it auto-calculate to match config defaults
        max_position_embeddings=512,
        use_moe=True,
        use_hierarchical_moe=True,
        num_l1_experts=4,
        num_l2_experts_per_group=4,
        num_experts_per_tok=2,
        l1_aux_loss_alpha=0.05,
        l2_aux_loss_alpha=0.05
    )
    
    print(f"Configuration: {config.hidden_size} hidden_size, {config.num_hidden_layers} layers")
    print(f"H-MoE: {config.num_l1_experts}x{config.num_l2_experts_per_group} experts")
    
    # Create model
    model = MiniMindForCausalLM(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Save checkpoint
    checkpoint_path = './out/pretrain_512_moe.pth'
    state_dict = {k: v.half() for k, v in model.state_dict().items()}
    torch.save(state_dict, checkpoint_path)
    
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Also save standard pretrain checkpoint
    checkpoint_path_std = './out/full_sft_512_moe.pth'  
    torch.save(state_dict, checkpoint_path_std)
    print(f"SFT checkpoint saved to: {checkpoint_path_std}")
    
    return True

if __name__ == "__main__":
    create_checkpoint()