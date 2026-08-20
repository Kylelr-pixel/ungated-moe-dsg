import torch
import time
import math
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator

# Benchmark Configuration
BATCH_SIZE = 32
SEQ_LEN = 128
HIDDEN_DIM = 256
SIGNATURE_DIM = 64
NUM_EXPERTS = 8
NUM_STEPS = 500

print("="*60)
print("STARTING INDUSTRY-STANDARD MoE KERNEL BENCHMARKS")
print("="*60)

# Initialize Models and Projections
dsg = DomainSignatureGenerator(HIDDEN_DIM, SIGNATURE_DIM)
expert_proj = torch.nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
expert_weights = torch.randn(NUM_EXPERTS, HIDDEN_DIM, requires_grad=True)

# Optimizer for Convergence Testing
optimizer = torch.optim.AdamW(
    list(dsg.parameters()) + list(expert_proj.parameters()) + [expert_weights], 
    lr=1e-3
)

# Metrics Tracking
expert_usage_counts = torch.zeros(NUM_EXPERTS)
grad_norms = []
start_time = time.time()

for step in range(1, NUM_STEPS + 1):
    optimizer.zero_grad()
    
    # Synthetic Input Data
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    
    # 1. Forward Pass through DSG
    sig_weights = dsg(x)
    gate_logits = expert_proj(sig_weights)
    gate_weights = torch.sigmoid(gate_logits)
    active_mask = (gate_weights > 0.5).float()
    
    # Record Routing Utilization
    with torch.no_grad():
        expert_usage_counts += active_mask.sum(dim=(0, 1))
    
    # 2. Kernel Execution
    expert_outputs = x.unsqueeze(2) * expert_weights
    out = ScaledUnGatedFunction.apply(
        expert_outputs, 
        gate_weights.unsqueeze(-1), 
        active_mask.unsqueeze(-1), 
        0.1
    )
    
    # Target loss: push output variance/energy toward 0
    loss = out.pow(2).mean()
    
    # 3. Backward Pass & Gradient Health
    loss.backward()
    
    # Track Gradient Norm of Expert Weights
    if expert_weights.grad is not None:
        grad_norms.append(expert_weights.grad.norm().item())
        
    optimizer.step()

total_time = time.time() - start_time

# --- EVALUATION METRICS REPORT ---
total_tokens_processed = BATCH_SIZE * SEQ_LEN * NUM_STEPS
tokens_per_sec = total_tokens_processed / total_time

# Load Balance CV (Coefficient of Variation)
mean_usage = expert_usage_counts.mean().item()
std_usage = expert_usage_counts.std().item()
cv = (std_usage / mean_usage) * 100 if mean_usage > 0 else 0

print("\n" + "="*20 + " BENCHMARK RESULTS " + "="*20)
print(f"Total Steps Completed : {NUM_STEPS}")
print(f"Total Time Elapsed    : {total_time:.2f} seconds")
print(f"Throughput Speed      : {tokens_per_sec:.2f} tokens/sec")
print(f"Final Step Loss       : {loss.item():.6f}")

print("\n[1] Expert Routing Load Distribution:")
for i, count in enumerate(expert_usage_counts):
    pct = (count / expert_usage_counts.sum()) * 100
    print(f"  - Expert {i}: {int(count.item()):>6} activations ({pct:.1f}%)")

print(f"\n[2] Load Balance Imbalance (CV): {cv:.2f}%")
if cv < 20:
    print("  -> STATUS: EXCELLENT (Even expert distribution)")
elif cv < 50:
    print("  -> STATUS: MODERATE (Slight expert preference)")
else:
    print("  -> STATUS: POOR (Routing imbalance detected)")

print("\n[3] Gradient Stability Check:")
avg_grad = sum(grad_norms) / len(grad_norms)
print(f"  - Average Expert Grad Norm : {avg_grad:.4f}")
print(f"  - Min/Max Grad Norm        : {min(grad_norms):.4f} / {max(grad_norms):.4f}")
if math.isnan(avg_grad) or max(grad_norms) > 100.0:
    print("  -> STATUS: WARNING (Potential exploding gradients)")
else:
    print("  -> STATUS: HEALTHY (Stable gradient propagation)")
print("="*60)
