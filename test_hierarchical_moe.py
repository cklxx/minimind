#!/usr/bin/env python3
"""
Test script for Hierarchical MoE implementation
"""

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def test_hierarchical_moe():
    print("Testing Hierarchical MoE implementation...")
    
    # Create config with hierarchical MoE enabled
    config = MiniMindConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=128,
        use_moe=True,
        use_hierarchical_moe=True,
        num_l1_experts=4,
        num_l2_experts_per_group=4,
        num_experts_per_tok=2,
        l1_aux_loss_alpha=0.05,
        l2_aux_loss_alpha=0.05
    )
    
    print("Config: L1 experts={}, L2 experts per group={}".format(
        config.num_l1_experts, config.num_l2_experts_per_group))
    print("Total experts: {}".format(config.num_l1_experts * config.num_l2_experts_per_group))
    
    # Create model
    model = MiniMindForCausalLM(config)
    model.eval()
    
    print("Model created with {} parameters".format(sum(p.numel() for p in model.parameters())))
    
    # Create test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print("Input shape: {}".format(input_ids.shape))
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids)
        
    print("Output logits shape: {}".format(output.logits.shape))
    print("Auxiliary loss: {}".format(output.aux_loss))
    
    # Test training mode
    model.train()
    output_train = model(input_ids)
    print("Training mode - Auxiliary loss: {}".format(output_train.aux_loss))
    
    # Check that hierarchical MoE layers are present
    hierarchical_layers = 0
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'l1_expert_groups'):
            hierarchical_layers += 1
            l1_groups = len(layer.mlp.l1_expert_groups)
            l2_experts = len(layer.mlp.l1_expert_groups[0].experts)
            print("Layer {}: {} L1 groups, {} L2 experts per group".format(
                layer.layer_id, l1_groups, l2_experts))
    
    print("Found {} hierarchical MoE layers".format(hierarchical_layers))
    
    print("Hierarchical MoE test completed successfully!")
    return True

if __name__ == "__main__":
    test_hierarchical_moe()