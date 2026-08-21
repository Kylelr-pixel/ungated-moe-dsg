import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledUnGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs, gate_weights, active_mask, scale_factor):
        ctx.save_for_backward(expert_outputs, gate_weights, active_mask)
        ctx.scale_factor = scale_factor
        return (expert_outputs * gate_weights * active_mask) * scale_factor

    @staticmethod
    def backward(ctx, grad_output):
        expert_outputs, gate_weights, active_mask = ctx.saved_tensors
        scale = ctx.scale_factor
        
        # 1. Primary gradient for active experts
        grad_active = grad_output * gate_weights * active_mask * scale
        
        # 2. Calibrated feedback for inactive experts
        inactive_mask = 1.0 - active_mask
        leakage_coefficient = 0.05
        grad_inactive = grad_output * gate_weights * inactive_mask * (scale * leakage_coefficient)
        
        # 3. Combined total gradient
        total_expert_grad = grad_active + grad_inactive
        grad_gate = (grad_output * expert_outputs * active_mask * scale).sum(dim=-1, keepdim=True)
        
        return total_expert_grad, grad_gate, None, None


class SwiGLUExpert(nn.Module):
    """Non-linear SwiGLU Feed-Forward Network Expert."""
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU activation: (Swish(x * W1) * (x * W2)) * W3
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DomainSignatureGenerator(nn.Module):
    def __init__(self, hidden_dim, signature_dim):
        super().__init__()
        self.signature_proj = nn.Sequential(
            nn.Linear(hidden_dim, signature_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.signature_proj(x)
