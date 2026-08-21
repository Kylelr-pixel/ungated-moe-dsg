import torch
import torch.nn as nn

class ScaledUnGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs, gate_weights, active_mask, scale_factor):
        ctx.save_for_backward(expert_outputs, gate_weights, active_mask)
        ctx.scale_factor = scale_factor
        
        # Forward pass executes sparse routing
        return (expert_outputs * gate_weights * active_mask) * scale_factor

    @staticmethod
    def backward(ctx, grad_output):
        expert_outputs, gate_weights, active_mask = ctx.saved_tensors
        scale = ctx.scale_factor
        
        # 1. Primary gradient for selected experts (active_mask == 1)
        grad_active = grad_output * gate_weights * active_mask * scale
        
        # 2. CALIBRATED BACKPROP FOR ZERO-EXPERTS (active_mask == 0)
        # Forces non-zero, scaled gradient feedback into inactive experts
        inactive_mask = 1.0 - active_mask
        leakage_coefficient = 0.05
        grad_inactive = grad_output * gate_weights * inactive_mask * (scale * leakage_coefficient)
        
        # 3. Combine active and inactive gradients
        total_expert_grad = grad_active + grad_inactive
        
        grad_gate = (grad_output * expert_outputs * active_mask * scale).sum(dim=-1, keepdim=True)
        
        return total_expert_grad, grad_gate, None, None


class DomainSignatureGenerator(nn.Module):
    def __init__(self, hidden_dim, signature_dim):
        super().__init__()
        self.signature_proj = nn.Sequential(
            nn.Linear(hidden_dim, signature_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.signature_proj(x)
