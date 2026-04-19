import torch
from open_mythos import OpenMythosForCausalLM, MythosConfig

def test_hf_generate():
    print("Initializing dummy HF model...")
    cfg = MythosConfig(
        vocab_size=1000, 
        dim=256, 
        n_heads=8, 
        max_loop_iters=4, 
        attn_type="mla",
        n_kv_heads=8,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16
    )
    model = OpenMythosForCausalLM(cfg)
    model.eval()

    # Create dummy prompt
    input_ids = torch.randint(0, cfg.vocab_size, (1, 10))
    
    print("Testing generate with max_new_tokens=5")
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=5)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 15), "Generation output shape is incorrect."
    
    print("HF Generate successfully utilized the OpenMythos past_key_values cache.")

if __name__ == "__main__":
    test_hf_generate()