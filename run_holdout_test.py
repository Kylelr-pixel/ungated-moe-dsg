import torch
import torch.nn as nn
import torch.nn.functional as F
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator, SwiGLUExpert

# --- SPARSE EXPLORATION CONFIG: Testing Top-K Sparsity & Calibration Handoffs ---
BATCH_SIZE = 32
SEQ_LEN = 128
HIDDEN_DIM = 256
INTERMEDIATE_DIM = 512
SIGNATURE_DIM = 64
NUM_EXPERTS = 8
TOP_K = 2  # We can adjust this to 1 or 4 to test sparsity stress!
STEPS_PER_DOMAIN = 150

print("="*85)
print(f"  SPARSE ROUTING STRESS TEST: Top-{TOP_K} out of {NUM_EXPERTS} Experts + Calibration Handoff")
print("="*85)

dsg = DomainSignatureGenerator(HIDDEN_DIM, SIGNATURE_DIM)
expert_proj = nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
experts = nn.ModuleList([SwiGLUExpert(HIDDEN_DIM, INTERMEDIATE_DIM) for _ in range(NUM_EXPERTS)])

dense_baseline = nn.Sequential(
    nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False),
    nn.SiLU(),
    nn.Linear(INTERMEDIATE_DIM, HIDDEN_DIM, bias=False)
)

optimizer = torch.optim.AdamW(
    list(dsg.parameters()) + list(expert_proj.parameters()) + list(experts.parameters()), 
    lr=1e-3
)
opt_dense = torch.optim.AdamW(dense_baseline.parameters(), lr=1e-3)

torch.manual_seed(101)
domains = {
    "Domain A (High-Variance Clusters)": torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 4.0,
    "Domain B (Sparse Spike Signal)": (torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) > 0.85).float() * 8.0,
    "Domain C (Low-Amplitude Smooth)": torch.sin(torch.linspace(0, 50, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM)).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 0.5,
    "Domain D (Out-of-Bounds Bimodal)": torch.bernoulli(torch.full((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), 0.5)) * 5.0 - 2.5
}

total_tokens_per_step = BATCH_SIZE * SEQ_LEN * TOP_K

for domain_name, data_tensor in domains.items():
    head_token_counts = torch.zeros(NUM_EXPERTS)
    head_active_grads = torch.zeros(NUM_EXPERTS)
    head_inactive_grads = torch.zeros(NUM_EXPERTS)
    
    for step in range(1, STEPS_PER_DOMAIN + 1):
        # 1. Dense Baseline Step
        opt_dense.zero_grad()
        out_dense = dense_baseline(data_tensor)
        loss_dense = F.mse_loss(out_dense, data_tensor)
        loss_dense.backward()
        opt_dense.step()
        
        # 2. Sparse ScaledUnGated MoE Step
        optimizer.zero_grad()
        sig_weights = dsg(data_tensor)
        gate_logits = expert_proj(sig_weights)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Sparse Top-K Selection
        topk_probs, topk_indices = torch.topk(gate_probs, k=TOP_K, dim=-1)
        active_mask = torch.zeros_like(gate_probs)
        active_mask.scatter_(-1, topk_indices, 1.0)
                
        with torch.no_grad():
            head_token_counts += active_mask.sum(dim=(0, 1))
            
        expert_outputs = torch.stack([expert(data_tensor) for expert in experts], dim=2)
        
        # Pass through ScaledUnGatedFunction (triggers calibration if mask is 0)
        out_moe_raw = ScaledUnGatedFunction.apply(
            expert_outputs, gate_probs.unsqueeze(-1), active_mask.unsqueeze(-1), 1.0
        )
        out_moe = out_moe_raw.sum(dim=2)
        
        primary_loss = F.mse_loss(out_moe, data_tensor)
        z_loss = torch.mean(torch.logsumexp(gate_logits, dim=-1)**2)
        
        tokens_per_exp = active_mask.sum(dim=(0, 1)) / total_tokens_per_step
        router_prob_per_exp = gate_probs.mean(dim=(0, 1))
        aux_loss = NUM_EXPERTS * torch.sum(tokens_per_exp * router_prob_per_exp)
        
        total_loss = primary_loss + (0.001 * z_loss) + (0.15 * aux_loss)
        total_loss.backward()
        
        # Track active vs calibration gradients explicitly
        with torch.no_grad():
            for i, expert in enumerate(experts):
                if expert.w1.weight.grad is not None:
                    g_norm = expert.w1.weight.grad.norm().item()
                    if active_mask[:, :, i].sum() > 0:
                        head_active_grads[i] += g_norm
                    else:
                        head_inactive_grads[i] += g_norm
                        
        optimizer.step()
        
    mean_u = head_token_counts.mean().item()
    std_u = head_token_counts.std().item()
    cv = (std_u / mean_u) * 100 if mean_u > 0 else 0
    
    print("\n" + "="*85)
    print(f"RESULTS FOR: {domain_name} (Top-{TOP_K} Sparse)")
    print("="*85)
    print(f"  - Dense Baseline MSE Loss : {loss_dense.item():.6f}")
    print(f"  - ScaledUnGated MoE MSE   : {primary_loss.item():.6f}")
    print(f"  - MSE Loss Reduction     : {((loss_dense.item() - primary_loss.item()) / loss_dense.item()) * 100:+.2f}%")
    print(f"  - Router Load Balance CV : {cv:.2f}% (Sparse Top-{TOP_K})")
    print("-" * 85)
    print(f"  PER-HEAD ACTIVITY & GRADIENT DISTRIBUTION OVER {STEPS_PER_DOMAIN} STEPS:")
    print(f"  {'Head ID':<8} | {'Tokens Routed':<16} | {'Avg Active Grad':<18} | {'Avg Inactive Grad':<18}")
    print("  " + "-" * 75)
    
    total_domain_tokens = head_token_counts.sum().item()
    for i in range(NUM_EXPERTS):
        cnt = int(head_token_counts[i].item())
        pct = (cnt / total_domain_tokens) * 100 if total_domain_tokens > 0 else 0
        act_g = head_active_grads[i].item() / STEPS_PER_DOMAIN
        inact_g = head_inactive_grads[i].item() / STEPS_PER_DOMAIN
        print(f"  Head {i:<3} | {cnt:<6} ({pct:5.1f}%) | {act_g:<18.6f} | {inact_g:<18.6f}")

print("="*85)
