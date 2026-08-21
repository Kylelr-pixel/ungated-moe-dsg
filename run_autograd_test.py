import torch
import math
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator

def test_autograd_sensitivity(scale_factor, lr, zero_mask=False):
    batch_size, seq_len, hidden_dim, signature_dim, num_experts = 32, 128, 256, 64, 8
    
    dsg = DomainSignatureGenerator(hidden_dim, signature_dim)
    expert_proj = torch.nn.Linear(signature_dim, num_experts)
    expert_weights = torch.randn(num_experts, hidden_dim, requires_grad=True)
    
    optimizer = torch.optim.AdamW(
        list(dsg.parameters()) + list(expert_proj.parameters()) + [expert_weights], 
        lr=lr
    )
    
    nan_detected = False
    inf_detected = False
    max_grad_norm = 0.0
    
    for step in range(1, 100):
        optimizer.zero_grad()
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        sig_weights = dsg(x)
        gate_logits = expert_proj(sig_weights)
        gate_weights = torch.sigmoid(gate_logits)
        
        if zero_mask:
            active_mask = torch.zeros_like(gate_weights)
        else:
            active_mask = (gate_weights > 0.5).float()
            
        expert_outputs = x.unsqueeze(2) * expert_weights
        out = ScaledUnGatedFunction.apply(
            expert_outputs, 
            gate_weights.unsqueeze(-1), 
            active_mask.unsqueeze(-1), 
            scale_factor
        )
        
        loss = out.pow(2).mean()
        loss.backward()
        
        # Check gradient status
        if expert_weights.grad is not None:
            grad_norm = expert_weights.grad.norm().item()
            if math.isnan(grad_norm):
                nan_detected = True
            elif math.isinf(grad_norm):
                inf_detected = True
            else:
                max_grad_norm = max(max_grad_norm, grad_norm)
                
        optimizer.step()
        
    status = "PASSED"
    if nan_detected:
        status = "FAILED (NaN Gradient)"
    elif inf_detected:
        status = "FAILED (Exploding Inf Gradient)"
        
    return max_grad_norm, status

print("="*65)
print("  SCALED-UNGATED AUTOGRAD KERNEL SENSITIVITY TEST")
print("="*65)

# Test Scenarios
scenarios = [
    ("Default Scaling (scale=0.1, lr=1e-3)", 0.1, 1e-3, False),
    ("High Scale Factor (scale=10.0, lr=1e-3)", 10.0, 1e-3, False),
    ("Micro Scale Factor (scale=1e-5, lr=1e-3)", 1e-5, 1e-3, False),
    ("Aggressive LR (scale=0.1, lr=1.0)", 0.1, 1.0, False),
    ("Zero Active Mask Edge Case", 0.1, 1e-3, True)
]

for label, scale, lr, mask_flag in scenarios:
    max_norm, status = test_autograd_sensitivity(scale, lr, zero_mask=mask_flag)
    print(f"  - {label:<40} | Max Grad Norm: {max_norm:>8.4f} | Status: {status}")

print("="*65)
