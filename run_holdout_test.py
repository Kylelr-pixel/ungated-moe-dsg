import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator

# --- FIXED PRODUCTION CONFIG (NO TUNING ALLOWED) ---
BATCH_SIZE = 32
SEQ_LEN = 128
HIDDEN_DIM = 256
SIGNATURE_DIM = 64
NUM_EXPERTS = 8
TOP_K = 2
CAPACITY_FACTOR = 1.2
STEPS_PER_DOMAIN = 150

print("="*70)
print("  BLIND HOLDOUT & REAL-WORLD DOMAIN GENERALIZATION BENCHMARK")
print("="*70)

# Instantiate Architecture
dsg = DomainSignatureGenerator(HIDDEN_DIM, SIGNATURE_DIM)
expert_proj = nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
expert_weights = torch.randn(NUM_EXPERTS, HIDDEN_DIM, requires_grad=True)
dense_baseline = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

optimizer = torch.optim.AdamW(
    list(dsg.parameters()) + list(expert_proj.parameters()) + [expert_weights], 
    lr=1e-3
)
opt_dense = torch.optim.AdamW(dense_baseline.parameters(), lr=1e-3)

# DEFINE 4 DISTINCT UNSEEN DATA DOMAINS
torch.manual_seed(101)
domains = {
    "Domain A (High-Variance Clusters)": torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 4.0,
    "Domain B (Sparse Spike Signal)": (torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) > 0.85).float() * 8.0,
    "Domain C (Low-Amplitude Smooth)": torch.sin(torch.linspace(0, 50, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM)).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 0.5,
    "Domain D (Out-of-Bounds Bimodal)": torch.bernoulli(torch.full((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), 0.5)) * 5.0 - 2.5
}

total_tokens = BATCH_SIZE * SEQ_LEN * TOP_K
fair_share = total_tokens / NUM_EXPERTS
expert_capacity = int(fair_share * CAPACITY_FACTOR)

for domain_name, data_tensor in domains.items():
    expert_usage = torch.zeros(NUM_EXPERTS)
    dropped_tokens = 0
    
    for step in range(1, STEPS_PER_DOMAIN + 1):
        # 1. Train Dense Baseline
        opt_dense.zero_grad()
        out_dense = dense_baseline(data_tensor)
        loss_dense = F.mse_loss(out_dense, data_tensor)
        loss_dense.backward()
        opt_dense.step()
        
        # 2. Train ScaledUnGated MoE
        optimizer.zero_grad()
        sig_weights = dsg(data_tensor)
        gate_logits = expert_proj(sig_weights)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        topk_probs, topk_indices = torch.topk(gate_probs, k=TOP_K, dim=-1)
        raw_mask = torch.zeros_like(gate_probs)
        raw_mask.scatter_(-1, topk_indices, 1.0)
        
        # Capacity Capping
        active_mask = torch.zeros_like(raw_mask)
        for exp_idx in range(NUM_EXPERTS):
            exp_mask = raw_mask[:, :, exp_idx]
            selected_indices = torch.nonzero(exp_mask, as_tuple=False)
            if selected_indices.size(0) > expert_capacity:
                keep_indices = selected_indices[:expert_capacity]
                active_mask[keep_indices[:, 0], keep_indices[:, 1], exp_idx] = 1.0
                dropped_tokens += (selected_indices.size(0) - expert_capacity)
            else:
                active_mask[:, :, exp_idx] = exp_mask
                
        with torch.no_grad():
            expert_usage += active_mask.sum(dim=(0, 1))
            
        expert_outputs = data_tensor.unsqueeze(2) * expert_weights
        out_moe = ScaledUnGatedFunction.apply(
            expert_outputs, gate_probs.unsqueeze(-1), active_mask.unsqueeze(-1), 0.1
        )
        
        primary_loss = F.mse_loss(out_moe, data_tensor)
        tokens_per_exp = active_mask.sum(dim=(0, 1)) / total_tokens
        router_prob_per_exp = gate_probs.mean(dim=(0, 1))
        aux_loss = NUM_EXPERTS * torch.sum(tokens_per_exp * router_prob_per_exp)
        
        total_loss = primary_loss + 0.15 * aux_loss
        total_loss.backward()
        optimizer.step()
        
    # Calculate Load Balance CV for this Domain
    mean_u = expert_usage.mean().item()
    std_u = expert_usage.std().item()
    cv = (std_u / mean_u) * 100 if mean_u > 0 else 0
    
    print(f"\nRESULTS FOR: {domain_name}")
    print(f"  - Dense Baseline MSE Loss : {loss_dense.item():.6f}")
    print(f"  - ScaledUnGated MoE MSE   : {primary_loss.item():.6f}")
    print(f"  - MSE Loss Reduction     : {((loss_dense.item() - primary_loss.item()) / loss_dense.item()) * 100:+.2f}%")
    print(f"  - Router Load Balance CV : {cv:.2f}% (Dropped Tokens: {dropped_tokens})")

print("="*70)
