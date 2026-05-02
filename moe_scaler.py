import torch
import torch.nn as nn
import torch.nn.functional as F
from open_mythos.main import MythosConfig, OpenMythos

class MoERouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.top_k = cfg.n_experts_per_tok
        self.router = nn.Linear(cfg.dim, self.n_experts, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(self.n_experts))

    def forward(self, x):
        logits = self.router(x) + self.gate_bias
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, self.top_k, dim=-1)
        shared_probs = torch.ones(x.size(0), self.n_shared, device=x.device) / self.n_shared
        shared_ids = torch.arange(self.n_shared, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        aux_loss = 0.01 * (probs.mean(0) * (probs.sum(0) / probs.size(0))).mean()
        return topk_ids, topk_probs, shared_ids, shared_probs, aux_loss

class MoEScalerAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.router = MoERouter(cfg)
        self.model = OpenMythos(cfg)
    
    def route_and_reason(self, task, n_loops=8):
        x = torch.randn(1, self.cfg.dim)
        topk_ids, topk_probs, shared_ids, shared_probs, aux = self.router(x)
        ids = torch.randint(0, self.cfg.vocab_size, (1, 10))
        out = self.model(ids, n_loops=n_loops)
        return f'MoE Routed: Top-K {topk_ids}, Shared {shared_ids}, Aux {aux:.4f}, Output {out.shape}.'

if __name__ == '__main__':
    cfg = mythos_1b()
    agent = MoEScalerAgent(cfg)
    print(agent.route_and_reason('test MoE mesh'))
 
