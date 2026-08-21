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

def run_telemetry_stress_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=========================================================")
    print(f"LONG-HORIZON TELEMETRY & GRADIENT STABILITY TEST ON: {device.upper()}")
    print(f"=========================================================\n")
    
    batch_size, seq_len, hidden_dim, num_experts, top_k = 4, 32, 64, 16, 2
    experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False).to(device) for _ in range(num_experts)])
    router = nn.Linear(hidden_dim, num_experts, bias=False).to(device)
    optimizer = torch.optim.AdamW(list(router.parameters()) + list(experts.parameters()), lr=1e-3)

    # Track gradient telemetry history across 20 continuous steps
    for step in range(1, 21):
        optimizer.zero_grad()
        inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device) * (1.0 + 0.05 * step)
        target_grad = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        logits = router(inputs)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        _, topk_idx = torch.topk(logits, k=top_k, dim=-1)
        active_mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0).unsqueeze(-1)
        
        expert_outputs = torch.stack([expert(inputs) for expert in experts], dim=-2)
        combined = conditional_moe_gate(expert_outputs, weights, active_mask, 1.0, 0.1)
        
        loss = (combined * target_grad).sum()
        loss.backward()
        
        # Calculate mean active vs skipped gradient norms
        active_norms = []
        skipped_norms = []
        for i, expert in enumerate(experts):
            norm = expert.weight.grad.norm().item() if expert.weight.grad is not None else 0.0
            is_active = active_mask[..., i, :].sum().item() > 0
            if is_active:
                active_norms.append(norm)
            else:
                skipped_norms.append(norm)
                
        mean_active = sum(active_norms) / len(active_norms) if active_norms else 0.0
        mean_skipped = sum(skipped_norms) / len(skipped_norms) if skipped_norms else 0.0
        
        print(f"Step {step:2d} | Active Heads Mean Grad Norm: {mean_active:8.4f} | Skipped Heads Mean Grad Norm: {mean_skipped:8.4f}")
        optimizer.step()

if __name__ == "__main__":
    run_telemetry_stress_test()
