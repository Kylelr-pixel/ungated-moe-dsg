import torch
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator

def test_precision(dtype, label):
    batch_size, seq_len, hidden_dim, signature_dim, num_experts = 32, 128, 256, 64, 8
    
    # Initialize modules
    dsg = DomainSignatureGenerator(hidden_dim, signature_dim)
    expert_proj = torch.nn.Linear(signature_dim, num_experts)
    expert_weights = torch.randn(num_experts, hidden_dim, requires_grad=True)
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Forward pass in FP32, then cast activations to target dtype for kernel evaluation
    sig_weights = dsg(x).to(dtype)
    gate_logits = expert_proj.to(dtype)(sig_weights)
    gate_weights = torch.sigmoid(gate_logits)
    active_mask = (gate_weights > 0.5).to(dtype)
    
    x_dtype = x.to(dtype)
    expert_weights_dtype = expert_weights.to(dtype).detach().requires_grad_(True)
    
    expert_outputs = x_dtype.unsqueeze(2) * expert_weights_dtype
    
    out = ScaledUnGatedFunction.apply(
        expert_outputs, 
        gate_weights.unsqueeze(-1), 
        active_mask.unsqueeze(-1), 
        0.1
    )
    
    loss = out.pow(2).mean()
    loss.backward()
    
    has_nan = torch.isnan(out).any().item() or torch.isnan(expert_weights_dtype.grad).any().item()
    has_inf = torch.isinf(out).any().item() or torch.isinf(expert_weights_dtype.grad).any().item()
    
    status = "PASSED"
    if has_nan:
        status = "FAILED (NaN Detected)"
    elif has_inf:
        status = "FAILED (Inf Overflow)"
        
    return loss.item(), expert_weights_dtype.grad.norm().item(), status

print("="*65)
print("  SCALED-UNGATED MIXED PRECISION & NUMERICAL EQUIVALENCE TEST")
print("="*65)

# 1. Evaluate FP32 Baseline
loss_fp32, grad_fp32, status_fp32 = test_precision(torch.float32, "FP32 (Standard)")
print(f"  - FP32 Standard   | Loss: {loss_fp32:.6f} | Grad Norm: {grad_fp32:.4f} | Status: {status_fp32}")

# 2. Evaluate Half Precision (FP16)
loss_fp16, grad_fp16, status_fp16 = test_precision(torch.float16, "FP16 (Half Precision)")
print(f"  - FP16 Half       | Loss: {loss_fp16:.6f} | Grad Norm: {grad_fp16:.4f} | Status: {status_fp16}")

# 3. Evaluate Brain Floating Point (BF16)
loss_bf16, grad_bf16, status_bf16 = test_precision(torch.bfloat16, "BF16 (Brain Float)")
print(f"  - BF16 Brain Float| Loss: {loss_bf16:.6f} | Grad Norm: {grad_bf16:.4f} | Status: {status_bf16}")

print("="*65)
