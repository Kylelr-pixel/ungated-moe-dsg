import torch
import time
import math
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator

def run_stress_test(num_experts, input_type="standard", steps=300):
    batch_size, seq_len, hidden_dim, signature_dim = 32, 128, 256, 64
    
    dsg = DomainSignatureGenerator(hidden_dim, signature_dim)
    expert_proj = torch.nn.Linear(signature_dim, num_experts)
    expert_weights = torch.randn(num_experts, hidden_dim, requires_grad=True)
    
    optimizer = torch.optim.AdamW(
        list(dsg.parameters()) + list(expert_proj.parameters()) + [expert_weights], 
        lr=1e-3
    )
    
    usage_counts = torch.zeros(num_experts)
    start_time = time.time()
    
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        
        # Inject Out-Of-Distribution (OOD) Skewed Inputs
        if input_type == "skewed":
            x = torch.randn(batch_size, seq_len, hidden_dim) * 5.0 + 2.5
        elif input_type == "sparse":
            x = (torch.rand(batch_size, seq_len, hidden_dim) > 0.8).float() * 10.0
        else:
            x = torch.randn(batch_size, seq_len, hidden_dim)
            
        sig_weights = dsg(x)
        gate_logits = expert_proj(sig_weights)
        gate_weights = torch.sigmoid(gate_logits)
        active_mask = (gate_weights > 0.5).float()
        
        with torch.no_grad():
            usage_counts += active_mask.sum(dim=(0, 1))
            
        expert_outputs = x.unsqueeze(2) * expert_weights
        out = ScaledUnGatedFunction.apply(
            expert_outputs, 
            gate_weights.unsqueeze(-1), 
            active_mask.unsqueeze(-1), 
            0.1
        )
        
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()
        
    elapsed = time.time() - start_time
    tokens_sec = (batch_size * seq_len * steps) / elapsed
    
    mean_u = usage_counts.mean().item()
    std_u = usage_counts.std().item()
    cv = (std_u / mean_u) * 100 if mean_u > 0 else 0
    
    return tokens_sec, cv, loss.item()

print("="*65)
print("  SCALED-UNGATED EXPERT SCALING & ROUTING STRESS TEST")
print("="*65)

# 1. Expert Count Scaling Test
print("\n[PHASE 1] EXPERT CAPACITY SCALING (Standard Gaussian Noise)")
for exp in [8, 16, 32]:
    tps, cv, final_loss = run_stress_test(num_experts=exp, input_type="standard")
    print(f"  - {exp:>2} Experts | Throughput: {tps:>8.1f} tok/s | Load CV: {cv:>5.2f}% | Loss: {final_loss:.4f}")

# 2. Out-Of-Distribution Input Stress Test
print("\n[PHASE 2] ROUTING COLLAPSE STRESS TEST (16 Experts)")
for dist in ["standard", "skewed", "sparse"]:
    tps, cv, final_loss = run_stress_test(num_experts=16, input_type=dist)
    status = "PASSED (Balanced)" if cv < 30 else "WARNING (Routing Bias)"
    print(f"  - Input: {dist:<9} | Load CV: {cv:>5.2f}% | Final Loss: {final_loss:.4f} | Status: {status}")

print("="*65)
