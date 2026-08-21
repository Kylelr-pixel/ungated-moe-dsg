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

def run_head_to_head_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=========================================================")
    print(f"HEAD-TO-HEAD MOE COMPARISON TEST ON: {device.upper()}")
    print(f"=========================================================\n")
    
    batch_size, seq_len, hidden_dim, num_experts = 4, 32, 128, 4
    inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
    target_grad = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    print("--- Testing Model A: Conditional Skipped-Head Calibration MoE ---")
    experts_a = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False).to(device) for _ in range(num_experts)])
    router_a = nn.Linear(hidden_dim, num_experts, bias=False).to(device)
    
    for iteration in range(3):
        optimizer = torch.optim.SGD(list(router_a.parameters()) + list(experts_a.parameters()), lr=0.01)
        optimizer.zero_grad()
        
        logits = router_a(inputs)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        _, topk_idx = torch.topk(logits, k=2, dim=-1)
        active_mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0).unsqueeze(-1)
        
        expert_outputs = torch.stack([expert(inputs) for expert in experts_a], dim=-2)
        combined = conditional_moe_gate(expert_outputs, weights, active_mask, 1.0, 0.1)
        
        loss = (combined * target_grad).sum()
        loss.backward()
        
        print(f"Iteration {iteration+1}:")
        for i, expert in enumerate(experts_a):
            head_tokens_assigned = active_mask[..., i, :].sum().item()
            grad_norm = expert.weight.grad.norm().item() if expert.weight.grad is not None else 0.0
            status = "ACTIVE" if head_tokens_assigned > 0 else "SKIPPED (Calibrated Gradient)"
            print(f"  Head {i}: Tokens = {int(head_tokens_assigned):3d} | Status = {status:30s} | Grad Norm = {grad_norm:.6f}")
        print("-" * 65)

    print("\n--- Testing Model B: Industry Standard Top-K MoE + L_aux ---")
    experts_b = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False).to(device) for _ in range(num_experts)])
    standard_router = StandardTopKMoERouter(hidden_dim, num_experts, top_k=2).to(device)
    
    for iteration in range(3):
        optimizer = torch.optim.SGD(list(standard_router.parameters()) + list(experts_b.parameters()), lr=0.01)
        optimizer.zero_grad()
        
        logits, topk_w, topk_idx, aux_loss = standard_router(inputs)
        
        out = torch.zeros_like(inputs)
        for b in range(batch_size):
            for s in range(seq_len):
                for k_idx in range(2):
                    expert_id = topk_idx[b, s, k_idx].item()
                    w = topk_w[b, s, k_idx]
                    out[b, s] += w * experts_b[expert_id](inputs[b, s])
                    
        total_loss = (out * target_grad).sum() + 0.01 * aux_loss
        total_loss.backward()
        
        print(f"Iteration {iteration+1} (Standard MoE + L_aux):")
        for i, expert in enumerate(experts_b):
            assigned_count = (topk_idx == i).sum().item()
            grad_norm = expert.weight.grad.norm().item() if expert.weight.grad is not None else 0.0
            status = "ACTIVE" if assigned_count > 0 else "STARVED (Zero Grad via Standard Truncation)"
            print(f"  Head {i}: Assignments = {assigned_count:3d} | Status = {status:40s} | Grad Norm = {grad_norm:.6f}")
        print(f"  Auxiliary Load Balancing Loss (L_aux): {aux_loss.item():.4f}")
        print("-" * 65)

if __name__ == "__main__":
    run_head_to_head_comparison()
