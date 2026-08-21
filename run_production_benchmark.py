import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator

# --- CONFIGURATION ---
BATCH_SIZE = 32
SEQ_LEN = 128
HIDDEN_DIM = 256
SIGNATURE_DIM = 64
NUM_EXPERTS = 8
TOP_K = 2
CAPACITY_FACTOR = 1.2  # Max 120% of fair-share tokens per expert
STEPS = 300

print("="*65)
print("  PRODUCTION MoE KERNEL VALIDATION (WITH CAPACITY CAPPING)")
print("="*65)

# 1. STRUCTURED SYNTHETIC CLUSTERS
torch.manual_seed(42)
cluster_centers = torch.randn(NUM_EXPERTS, HIDDEN_DIM) * 3.0
cluster_ids = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE * SEQ_LEN,))
structured_x = cluster_centers[cluster_ids] + torch.randn(BATCH_SIZE * SEQ_LEN, HIDDEN_DIM) * 0.5
structured_x = structured_x.view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

# 2. MODULE INITIALIZATION
dsg = DomainSignatureGenerator(HIDDEN_DIM, SIGNATURE_DIM)
expert_proj = nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
expert_weights = torch.randn(NUM_EXPERTS, HIDDEN_DIM, requires_grad=True)

optimizer = torch.optim.AdamW(
    list(dsg.parameters()) + list(expert_proj.parameters()) + [expert_weights], 
    lr=1e-3
)

# Calculate Expert Token Capacity Limit
total_tokens = BATCH_SIZE * SEQ_LEN * TOP_K
fair_share = total_tokens / NUM_EXPERTS
expert_capacity = int(fair_share * CAPACITY_FACTOR)

expert_usage_counts = torch.zeros(NUM_EXPERTS)
dropped_tokens_total = 0

start_moe = time.time()

for step in range(1, STEPS + 1):
    optimizer.zero_grad()
    
    # Forward Pass through DSG
    sig_weights = dsg(structured_x)
    gate_logits = expert_proj(sig_weights)
    
    # Add Router Noise Injection during training to encourage exploration
    noise = torch.randn_like(gate_logits) * 0.1
    gate_probs = F.softmax(gate_logits + noise, dim=-1)
    
    # Top-k Expert Selection
    topk_probs, topk_indices = torch.topk(gate_probs, k=TOP_K, dim=-1)
    
    # Create Raw Active Mask
    raw_active_mask = torch.zeros_like(gate_probs)
    raw_active_mask.scatter_(-1, topk_indices, 1.0)
    
    # Enforce Capacity Limit per Expert
    active_mask = torch.zeros_like(raw_active_mask)
    for exp_idx in range(NUM_EXPERTS):
        exp_mask = raw_active_mask[:, :, exp_idx]
        selected_indices = torch.nonzero(exp_mask, as_tuple=False)
        
        if selected_indices.size(0) > expert_capacity:
            # Cap activations at max capacity
            keep_indices = selected_indices[:expert_capacity]
            active_mask[keep_indices[:, 0], keep_indices[:, 1], exp_idx] = 1.0
            dropped_tokens_total += (selected_indices.size(0) - expert_capacity)
        else:
            active_mask[:, :, exp_idx] = exp_mask
            
    with torch.no_grad():
        expert_usage_counts += active_mask.sum(dim=(0, 1))
    
    # Kernel Execution
    expert_outputs = structured_x.unsqueeze(2) * expert_weights
    out = ScaledUnGatedFunction.apply(
        expert_outputs, 
        gate_probs.unsqueeze(-1), 
        active_mask.unsqueeze(-1), 
        0.1
    )
    
    # Primary Loss + Strong Auxiliary Loss (0.15 weight)
    primary_loss = out.pow(2).mean()
    tokens_per_expert = active_mask.sum(dim=(0, 1)) / total_tokens
    router_prob_per_expert = gate_probs.mean(dim=(0, 1))
    aux_loss = NUM_EXPERTS * torch.sum(tokens_per_expert * router_prob_per_expert)
    
    total_loss = primary_loss + 0.15 * aux_loss
    total_loss.backward()
    optimizer.step()

time_moe = time.time() - start_moe
tps_moe = (BATCH_SIZE * SEQ_LEN * STEPS) / time_moe

# Load Balance Metrics
mean_u = expert_usage_counts.mean().item()
std_u = expert_usage_counts.std().item()
cv = (std_u / mean_u) * 100 if mean_u > 0 else 0

print("\n[1] ROUTING & CAPACITY RESULTS")
print(f"  - Load Balance CV      : {cv:.2f}%")
print(f"  - Total Dropped Tokens : {dropped_tokens_total}")
print(f"  - Throughput Speed     : {tps_moe:.1f} tok/s")

print("\n[2] EXPERT TOKEN DISTRIBUTION")
for i, count in enumerate(expert_usage_counts):
    pct = (count / expert_usage_counts.sum()) * 100
    print(f"    * Expert {i}: {int(count.item()):>6} tokens ({pct:.1f}%)")

print("\n[3] LOSS METRICS")
print(f"  - Primary Loss         : {primary_loss.item():.6f}")
print(f"  - Auxiliary Loss       : {aux_loss.item():.6f}")
print("="*65)
