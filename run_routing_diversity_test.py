import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineTopKOrchestrator(nn.Module):
    def __init__(self, hidden_dim, signature_dim, num_experts, top_k=2, temperature=0.2):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.signature_proj = nn.Sequential(
            nn.Linear(hidden_dim, signature_dim),
            nn.LayerNorm(signature_dim),
            nn.GELU(),
            nn.Linear(signature_dim, signature_dim)
        )
        self.expert_signatures = nn.Parameter(torch.empty(num_experts, signature_dim))
        nn.init.orthogonal_(self.expert_signatures)

    def forward(self, x):
        domain_sigs = F.normalize(self.signature_proj(x.float()), p=2, dim=-1)
        norm_expert_sigs = F.normalize(self.expert_signatures.float(), p=2, dim=-1)
        logits = torch.matmul(domain_sigs, norm_expert_sigs.T) / self.temperature
        return logits

    def compute_ortho_loss(self):
        norm_sigs = F.normalize(self.expert_signatures.float(), p=2, dim=-1)
        identity = torch.eye(norm_sigs.size(0), device=norm_sigs.device)
        return torch.norm(torch.matmul(norm_sigs, norm_sigs.T) - identity)

def run_routing_diversity_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=========================================================")
    print(f"ROUTING COLLAPSE & EXPERT DIVERSITY STRESS TEST ON: {device.upper()}")
    print(f"=========================================================\n")
    
    batch_size, seq_len, hidden_dim, num_experts, top_k = 4, 32, 64, 16, 2
    router = CosineTopKOrchestrator(hidden_dim=hidden_dim, signature_dim=16, num_experts=num_experts, top_k=top_k).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)

    for step in range(1, 11):
        optimizer.zero_grad()
        inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        logits = router(inputs)
        probs = F.softmax(logits, dim=-1)
        
        # Calculate routing probability entropy (higher entropy = better diversity, no collapse)
        avg_probs = probs.mean(dim=(0, 1))
        entropy = -(avg_probs * torch.log(avg_probs + 1e-9)).sum().item()
        max_entropy = math.log(num_experts)
        diversity_score = entropy / max_entropy
        
        ortho_loss = router.compute_ortho_loss()
        loss = ortho_loss - 0.1 * entropy  # maximize entropy while keeping signatures orthogonal
        
        loss.backward()
        optimizer.step()
        
        print(f"Step {step:2d} | Routing Entropy: {entropy:6.4f} | Diversity Ratio: {diversity_score:6.4f} | Ortho Loss: {ortho_loss.item():6.4f}")

if __name__ == "__main__":
    run_routing_diversity_test()
