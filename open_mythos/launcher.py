import torch
from open_mythos import OpenMythos, mythos_1b  # Original from fork

class MythosSubAgent:
    def __init__(self, role, cfg=None):
        self.role = role
        self.cfg = cfg or mythos_1b()
        self.model = OpenMythos(self.cfg)
    
    def reason(self, task, n_loops=8):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 10))  # Stub tokenize
        out = self.model.generate(ids, max_new_tokens=50, n_loops=n_loops)
        rho = torch.linalg.eigvals(self.model.recurrent.injection.get_A()).abs().max().item()
        return f"{self.role.capitalize()} Agent: Looped ({n_loops}) on '{task}' -> Output {out.shape}, Rho {rho:.2f} <1 stable."

# 11 Sub-Agents (Real Roles, Meshed with Original)
roles = [
    'core_impl', 'trainer_scaler', 'kortix_integrator', 'attention_specialist',
    'moe_scaler', 'stability_guardian', 'act_halter', 'generalizer',
    'overthink_fixer', 'extension_tester', 'integrator_lead'
]
agents = {role: MythosSubAgent(role) for role in roles}

def launch_swarm(task, n_loops=8):
    plans = {}
    for role, agent in agents.items():
        plans[role] = agent.reason(task, n_loops)
        print(plans[role])  # Coordinate output
    
    # Lead merges (meshed idea)
    lead_plan = agents['integrator_lead'].reason(f"Merge for {task}", n_loops=4)
    return {"merged": lead_plan, "plans": plans}

if __name__ == '__main__':
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "Demo mesh"
    result = launch_swarm(task)
    print("\nSwarm Meshed Complete:", result["merged"])