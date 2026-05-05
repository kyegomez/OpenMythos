import torch
from open_mythos.moda import MoDAConfig, MoDAModel


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tiny config: 4 layers, 8 routed experts, top-2
    cfg = MoDAConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=32,
        max_seq_len=64,
        # MoE: 2 shared + 8 routed, activate top-2
        # (2+2)*64 = 256 ≈ equivalent to dense SwiGLU hidden~256
        n_shared_experts=2,
        n_routed_experts=8,
        n_activated_experts=2,
        expert_hidden_dim=64,
        moe_balance_alpha=0.01,
        moe_score_func="softmax",
    )

    model = MoDAModel(cfg).to(device)
    print(f"Parameters: {model.num_parameters():,}")
    print(model)

    B, T = 2, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    labels = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    logits, loss = model(input_ids, labels)
    assert logits.shape == (B, T, cfg.vocab_size)
    print(f"Logits shape : {logits.shape}")
    print(f"Loss (LM + balance): {loss.item():.4f}")

    loss.backward()

    # Verify gradients
    last_writes = {
        f"blocks.{cfg.n_layers - 1}.k_write.weight",
        f"blocks.{cfg.n_layers - 1}.v_write.weight",
    }
    missing = [
        name
        for name, p in model.named_parameters()
        if p.grad is None and name not in last_writes
    ]
    if missing:
        print(f"WARNING — unexpected missing gradients: {missing}")
    else:
        print("All parameters received gradients (excluding last-block writes).")

    # Spot-check: MoE gate weights must receive gradients (through balance loss P_i)
    gate0_grad = model.blocks[0].moe.gate.weight.grad
    assert gate0_grad is not None, "blocks[0].moe.gate.weight has no gradient!"
    print(f"blocks[0].moe.gate.weight grad norm : {gate0_grad.norm().item():.6f}")

    # Spot-check: depth write projections gradient flows from layer ≥ 1 depth reads
    k0_grad = model.blocks[0].k_write.weight.grad
    assert k0_grad is not None, "blocks[0].k_write.weight has no gradient!"
    print(f"blocks[0].k_write.weight grad norm  : {k0_grad.norm().item():.6f}")

    print("Smoke test passed.")
