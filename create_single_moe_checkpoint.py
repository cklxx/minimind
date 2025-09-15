#!/usr/bin/env python3
"""
Create a test checkpoint for single-layer MoE
"""

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def create_checkpoint():
    print("Creating single-layer MoE test checkpoint...")
    
    # Create config with single-layer MoE
    config = MiniMindConfig(
        vocab_size=6400,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=None,  # Auto-calculate
        max_position_embeddings=512,
        use_moe=True,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1
    )
    
    print(f"Configuration: {config.hidden_size} hidden_size, {config.num_hidden_layers} layers")
    print(f"MoE: {config.n_routed_experts} experts, {config.num_experts_per_tok} per token")
    
    # Create model
    model = MiniMindForCausalLM(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Save checkpoint
    checkpoint_path = './out/pretrain_512_moe.pth'
    state_dict = {k: v.half() for k, v in model.state_dict().items()}
    torch.save(state_dict, checkpoint_path)
    
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Also save standard pretrain and SFT checkpoints
    checkpoint_path_sft = './out/full_sft_512_moe.pth'  
    torch.save(state_dict, checkpoint_path_sft)
    print(f"SFT checkpoint saved to: {checkpoint_path_sft}")
    
    return True

if __name__ == "__main__":
    create_checkpoint()