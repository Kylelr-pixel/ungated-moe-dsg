import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator

# --- CONFIGURATION ---
BATCH_SIZE = 32
SEQ_LEN = 128
HIDDEN_DIM = 256
SIGNATURE_DIM = 64
NUM_EXPERTS = 8
TOP_K = 2
STEPS = 300

print("="*65)
print("  PRODUCTION MoE KERNEL VALIDATION & BASELINE COMPARISON")
print("="*65)

# 1. GENERATE STRUCTURED SYNTHETIC CLUSTERS (Not Pure Noise)
torch.manual_seed(42)
cluster_centers = torch.randn(NUM_EXPERTS, HIDDEN_DIM) * 3.0
cluster_ids = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE * SEQ_LEN,))
structured_x = cluster_centers[cluster_ids] + torch.randn(BATCH_SIZE * SEQ_LEN, HIDDEN_DIM) * 0.5
structured_x = structured_x.view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

# 2. DEFINE BASELINE DENSE LAYER
dense_layer = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

# 3. DEFINE SCALED-UNGATED MOE MODULE
dsg = DomainSignatureGenerator(HIDDEN_DIM, SIGNATURE_DIM)
expert_proj = nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
expert_weights = torch.randn(NUM_EXPERTS, HIDDEN_DIM, requires_grad=True)

optimizer = torch.optim.AdamW(
    list(dsg.parameters()) + list(expert_proj.parameters()) + [expert_weights], 
    lr=1e-3
)

# --- BENCHMARK 1: DENSE BASELINE SPEED ---
start_dense = time.time()
for _ in range(STEPS):
    out_dense = dense_layer(structured_x)
    loss_dense = out_dense.pow(2).mean()
    loss_dense.backward()
time_dense = time.time() - start_dense
tps_dense = (BATCH_SIZE * SEQ_LEN * STEPS) / time_dense

# --- BENCHMARK 2: TOP-2 ROUTING WITH AUXILIARY LOSS ---
expert_usage_counts = torch.zeros(NUM_EXPERTS)
start_moe = time.time()

for step in range(1, STEPS + 1):
    optimizer.zero_grad()
    
    # DSG Forward Pass
    sig_weights = dsg(structured_x)
    gate_logits = expert_proj(sig_weights)
    gate_probs = F.softmax(gate_logits, dim=-1)
    
    # Top-k Expert Selection (Industry Standard)
    topk_probs, topk_indices = torch.topk(gate_probs, k=TOP_K, dim=-1)
    
    # Create Sparse Active Mask
    active_mask = torch.zeros_like(gate_probs)
    active_mask.scatter_(-1, topk_indices, 1.0)
    
    with torch.no_grad():
        expert_usage_counts += active_mask.sum(dim=(0, 1))
    
    # Compute Custom Kernel
    expert_outputs = structured_x.unsqueeze(2) * expert_weights
    out = ScaledUnGatedFunction.apply(
        expert_outputs, 
        gate_probs.unsqueeze(-1), 
        active_mask.unsqueeze(-1), 
        0.1
    )
    
    # Primary Loss + Switch Transformer Load Balancing Aux Loss
    primary_loss = out.pow(2).mean()
    tokens_per_expert = active_mask.sum(dim=(0, 1)) / (BATCH_SIZE * SEQ_LEN * TOP_K)
    router_prob_per_expert = gate_probs.mean(dim=(0, 1))
    aux_loss = NUM_EXPERTS * torch.sum(tokens_per_expert * router_prob_per_expert)
    
    total_loss = primary_loss + 0.01 * aux_loss
    total_loss.backward()
    optimizer.step()

time_moe = time.time() - start_moe
tps_moe = (BATCH_SIZE * SEQ_LEN * STEPS) / time_moe

# Load Balance Metrics
mean_u = expert_usage_counts.mean().item()
std_u = expert_usage_counts.std().item()
cv = (std_u / mean_u) * 100 if mean_u > 0 else 0

# --- REPORT ---
print("\n[1] THROUGHPUT COMPARISON")
print(f"  - Dense Baseline Speed : {tps_dense:>10.1f} tok/s")
print(f"  - ScaledUnGated MoE    : {tps_moe:>10.1f} tok/s")
print(f"  - Relative Efficiency  : {(tps_moe / tps_dense) * 100:.1f}% of Dense speed")

print("\n[2] TOP-2 ROUTING LOAD BALANCE (Structured Data + Aux Loss)")
print(f"  - Load Balance CV      : {cv:.2f}%")
for i, count in enumerate(expert_usage_counts):
    pct = (count / expert_usage_counts.sum()) * 100
    print(f"    * Expert {i}: {int(count.item()):>6} tokens ({pct:.1f}%)")

print("\n[3] LOSS CONVERGENCE")
print(f"  - Final Primary Loss   : {primary_loss.item():.6f}")
print(f"  - Final Aux Loss       : {aux_loss.item():.6f}")
print("="*65)
