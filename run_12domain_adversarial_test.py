import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalSkippedMoEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs, gate_weights, active_mask, scale_factor, gamma=0.1):
        ctx.save_for_backward(expert_outputs, gate_weights, active_mask)
        ctx.scale_factor = scale_factor
        ctx.gamma = gamma
        return (expert_outputs * gate_weights * active_mask).sum(dim=-2) * scale_factor

    @staticmethod
    def backward(ctx, grad_output):
        expert_outputs, gate_weights, active_mask = ctx.saved_tensors
        scale = ctx.scale_factor
        gamma = ctx.gamma
        
        grad_out_expanded = grad_output.unsqueeze(-2)
        grad_active = grad_out_expanded * gate_weights * active_mask * scale
        
        inactive_mask = 1.0 - active_mask
        grad_skipped = grad_out_expanded * gate_weights * inactive_mask * (scale * gamma)
        
        total_expert_grad = grad_active + grad_skipped
        grad_gate = (grad_out_expanded * expert_outputs * active_mask * scale).sum(dim=-1, keepdim=True)
        
        return total_expert_grad, grad_gate, None, None, None

conditional_moe_gate = ConditionalSkippedMoEFunction.apply

class StandardTopKMoERouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, k=self.top_k, dim=-1)
        
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=(0, 1, 2))
        fraction_tokens = tokens_per_expert / expert_mask.shape[0] / expert_mask.shape[1] * self.top_k
        
        router_prob_per_expert = weights.mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(fraction_tokens * router_prob_per_expert)
        
        return logits, topk_weights, topk_indices, aux_loss

def run_12domain_stress_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=========================================================")
    print(f"12-DOMAIN ADVERSARIAL STRESS & CALIBRATION TEST ON: {device.upper()}")
    print(f"=========================================================\n")
    
    batch_size, seq_len, hidden_dim, num_experts, top_k = 2, 16, 64, 16, 2
    domains = [f"Domain_{i+1:02d}" for i in range(12)]
    
    experts_a = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False).to(device) for _ in range(num_experts)])
    router_a = nn.Linear(hidden_dim, num_experts, bias=False).to(device)
    
    experts_b = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False).to(device) for _ in range(num_experts)])
    standard_router = StandardTopKMoERouter(hidden_dim, num_experts, top_k=top_k).to(device)
    
    optimizer_a = torch.optim.SGD(list(router_a.parameters()) + list(experts_a.parameters()), lr=0.01)
    optimizer_b = torch.optim.SGD(list(standard_router.parameters()) + list(experts_b.parameters()), lr=0.01)

    for d_idx, domain_name in enumerate(domains):
        print(f"---------------------------------------------------------")
        print(f"TESTING ADVERSARIAL STREAM: {domain_name}")
        print(f"---------------------------------------------------------")
        
        torch.manual_seed(42 + d_idx)
        inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device) * (0.5 + 0.1 * d_idx) + (d_idx % 3)
        target_grad = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        # Model A
        optimizer_a.zero_grad()
        logits_a = router_a(inputs)
        weights_a = F.softmax(logits_a, dim=-1).unsqueeze(-1)
        _, topk_idx_a = torch.topk(logits_a, k=top_k, dim=-1)
        active_mask_a = torch.zeros_like(logits_a).scatter_(-1, topk_idx_a, 1.0).unsqueeze(-1)
        
        expert_outputs_a = torch.stack([expert(inputs) for expert in experts_a], dim=-2)
        combined_a = conditional_moe_gate(expert_outputs_a, weights_a, active_mask_a, 1.0, 0.1)
        loss_a = (combined_a * target_grad).sum()
        loss_a.backward()
        
        # Model B
        optimizer_b.zero_grad()
        logits_b, topk_w_b, topk_idx_b, aux_loss_b = standard_router(inputs)
        
        out_b = torch.zeros_like(inputs)
        for b in range(batch_size):
            for s in range(seq_len):
                for k_idx in range(top_k):
                    expert_id = topk_idx_b[b, s, k_idx].item()
                    w = topk_w_b[b, s, k_idx]
                    out_b[b, s] += w * experts_b[expert_id](inputs[b, s])
                    
        total_loss_b = (out_b * target_grad).sum() + 0.01 * aux_loss_b
        total_loss_b.backward()
        
        skipped_count_a = (active_mask_a.sum(dim=(0, 1, 3)) == 0).sum().item()
        starved_count_b = sum(1 for i in range(num_experts) if (topk_idx_b == i).sum().item() == 0)
        
        print(f"[{domain_name}] Model A (Calibration MoE): {skipped_count_a} heads SKIPPED (received calibrated grad)")
        print(f"[{domain_name}] Model B (Standard + L_aux): {starved_count_b} heads STARVED (zero grad)")
        
        print("  Model A Head Telemetry:")
        for i, expert in enumerate(experts_a):
            tokens = int(active_mask_a[..., i, :].sum().item())
            norm = expert.weight.grad.norm().item() if expert.weight.grad is not None else 0.0
            status = "ACTIVE" if tokens > 0 else "SKIPPED (Calibrated)"
            print(f"    Head {i:2d}: Tokens = {tokens:2d} | Status = {status:22s} | Grad Norm = {norm:.4f}")
        print("-" * 57)

if __name__ == "__main__":
    run_12domain_stress_test()
