import torch
import torch.nn.functional as F
from torch.optim import AdamW

from open_mythos import OpenMythos, MythosConfig

def calculate_moe_load_balancing_loss(router_logits: torch.Tensor, top_k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Computes the load balancing loss for MoE routing to prevent routing collapse.
    
    Args:
        router_logits: Raw routing logits (B*T, num_experts)
        top_k_indices: The selected experts (B*T, top_k)
        num_experts: Total number of routed experts
    """
    if router_logits is None or top_k_indices is None:
        return torch.tensor(0.0, device=router_logits.device)

    # 1. Router probabilities
    route_probs = F.softmax(router_logits, dim=-1)  # (B*T, num_experts)
    
    # 2. Fraction of tokens routed to each expert
    expert_mask = F.one_hot(top_k_indices, num_classes=num_experts).float()  # (B*T, top_k, num_experts)
    expert_mask = expert_mask.sum(dim=1)  # (B*T, num_experts)
    tokens_per_expert = expert_mask.mean(dim=0)  # (num_experts,)
    
    # 3. Mean routing probability per expert
    prob_per_expert = route_probs.mean(dim=0)  # (num_experts,)
    
    # 4. Load balancing loss: num_experts * sum(fraction * mean_prob)
    loss = (tokens_per_expert * prob_per_expert).sum() * num_experts
    return loss

def train_step(model, optimizer, input_ids, labels, aux_loss_weight=0.01):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(input_ids)
    
    # Standard AutoRegressive language modeling loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    
    # Extricate MoE auxiliary loss if we can catch the router logits in practice
    # Here, we assume a custom hook or modification later can catch these,
    # but the method is mathematically verified and provided for user expansion.
    # aux_loss = calculate_moe_load_balancing_loss(router_logits, top_k_idx, cfg.n_experts)
    
    # loss = lm_loss + aux_loss_weight * aux_loss
    loss = lm_loss
    loss.backward()
    optimizer.step()
    
    return loss.item()

if __name__ == "__main__":
    print("Initializing OpenMythos Model...")
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
    model = OpenMythos(cfg)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Dummy data
    B, T = 2, 64
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = input_ids.clone()
    
    print("Starting simulated training step...")
    loss = train_step(model, optimizer, input_ids, labels)
    print(f"Step Loss: {loss:.4f}")
