#!/usr/bin/env python3
"""
Test inference script for Hierarchical MoE model
"""

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def test_inference():
    print("Testing Hierarchical MoE Inference...")
    
    # Create config matching our training setup
    config = MiniMindConfig(
        vocab_size=6400,
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
    
    print("Configuration:")
    print(f"  - L1 experts: {config.num_l1_experts}")
    print(f"  - L2 experts per group: {config.num_l2_experts_per_group}")
    print(f"  - Total experts: {config.num_l1_experts * config.num_l2_experts_per_group}")
    print(f"  - Experts per token: {config.num_experts_per_tok}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model')
    
    # Create model
    model = MiniMindForCausalLM(config)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test input text
    test_texts = [
        "人工智能是",
        "深度学习的特点",
        "机器学习算法",
    ]
    
    print("\nTesting Inference:")
    print("-" * 50)
    
    with torch.no_grad():
        for i, text in enumerate(test_texts, 1):
            print(f"Test {i}: Input = '{text}'")
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # Generate
            output = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Output = '{generated_text}'")
            print()
    
    print("Testing hierarchical MoE structure:")
    print("-" * 50)
    
    # Check hierarchical structure
    hierarchical_layers = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'l1_expert_groups'):
            hierarchical_layers += 1
            l1_groups = len(layer.mlp.l1_expert_groups)
            l2_experts = len(layer.mlp.l1_expert_groups[0].experts) if l1_groups > 0 else 0
            print(f"Layer {layer_idx}: {l1_groups} L1 groups, {l2_experts} L2 experts per group")
    
    print(f"Total hierarchical MoE layers: {hierarchical_layers}")
    
    # Test with a simple forward pass to check auxiliary loss
    print("\nTesting forward pass with aux loss:")
    test_input = torch.randint(0, config.vocab_size, (1, 10))
    output = model(test_input)
    print(f"Logits shape: {output.logits.shape}")
    print(f"Auxiliary loss: {output.aux_loss}")
    
    print("\nHierarchical MoE inference test completed successfully! ✓")
    return True

if __name__ == "__main__":
    test_inference()