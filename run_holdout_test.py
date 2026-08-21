import torch
import torch.nn as nn
import torch.nn.functional as F
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator, SwiGLUExpert

# --- SIDE-BY-SIDE INDUSTRY STANDARD COMPARISON CONFIG ---
BATCH_SIZE = 16
SEQ_LEN = 64
HIDDEN_DIM = 256
INTERMEDIATE_DIM = 512
SIGNATURE_DIM = 64
NUM_EXPERTS = 16
TOP_K = 2
STEPS_PER_DOMAIN = 60

print("="*95)
print(f"  SIDE-BY-SIDE BENCHMARK: Standard Sparse MoE vs. ScaledUnGated Calibration MoE")
print(f"  ({NUM_EXPERTS} Experts, Top-{TOP_K} Sparse, 8 Unconstrained Domains)")
print("="*95)

torch.manual_seed(101)
domains = {
    "Domain A (High-Variance Clusters)": torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 4.0,
    "Domain B (Sparse Spike Signal)": (torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) > 0.85).float() * 8.0,
    "Domain C (Low-Amplitude Smooth Sine)": torch.sin(torch.linspace(0, 50, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM)).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 0.5,
    "Domain D (Out-of-Bounds Bimodal)": torch.bernoulli(torch.full((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), 0.5)) * 5.0 - 2.5,
    "Domain E (High-Freq Sine Oscillation)": torch.sin(torch.linspace(0, 200, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM)).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 2.0,
    "Domain F (Low-Amplitude Smooth Cosine)": torch.cos(torch.linspace(0, 50, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM)).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 0.5,
    "Domain G (Uniform Random Noise)": torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 6.0 - 3.0,
    "Domain H (Intermittent Step Function)": torch.sign(torch.sin(torch.linspace(0, 50, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM))).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 3.0
}

total_tokens_per_step = BATCH_SIZE * SEQ_LEN * TOP_K

for domain_name, data_tensor in domains.items():
    # --- MODEL A: Standard MoE (No Calibration - unselected experts get 0 grad) ---
    torch.manual_seed(42)
    dsg_std = DomainSignatureGenerator(HIDDEN_DIM, SIGNATURE_DIM)
    proj_std = nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
    experts_std = nn.ModuleList([SwiGLUExpert(HIDDEN_DIM, INTERMEDIATE_DIM) for _ in range(NUM_EXPERTS)])
    opt_std = torch.optim.AdamW(list(dsg_std.parameters()) + list(proj_std.parameters()) + list(experts_std.parameters()), lr=1e-3)

    # --- MODEL B: Your Calibrated MoE (ScaledUnGatedFunction) ---
    torch.manual_seed(42)
    dsg_cal = DomainSignatureGenerator(HIDDEN_DIM, SIGNATURE_DIM)
    proj_cal = nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
    experts_cal = nn.ModuleList([SwiGLUExpert(HIDDEN_DIM, INTERMEDIATE_DIM) for _ in range(NUM_EXPERTS)])
    opt_cal = torch.optim.AdamW(list(dsg_cal.parameters()) + list(proj_cal.parameters()) + list(experts_cal.parameters()), lr=1e-3)

    std_tokens = torch.zeros(NUM_EXPERTS)
    cal_tokens = torch.zeros(NUM_EXPERTS)

    for step in range(1, STEPS_PER_DOMAIN + 1):
        # --- Train Standard Model ---
        opt_std.zero_grad()
        logits_std = proj_std(dsg_std(data_tensor))
        probs_std = F.softmax(logits_std, dim=-1)
        topk_p_std, topk_i_std = torch.topk(probs_std, k=TOP_K, dim=-1)
        mask_std = torch.zeros_like(probs_std).scatter_(-1, topk_i_std, 1.0)
        
        with torch.no_grad():
            std_tokens += mask_std.sum(dim=(0, 1))

        expert_outs_std = torch.stack([exp(data_tensor) for exp in experts_std], dim=2)
        out_std = (expert_outs_std * mask_std.unsqueeze(-1)).sum(dim=2)
        loss_std = F.mse_loss(out_std, data_tensor) + 0.001 * torch.mean(torch.logsumexp(logits_std, dim=-1)**2)
        loss_std.backward()
        opt_std.step()

        # --- Train Calibrated Model ---
        opt_cal.zero_grad()
        logits_cal = proj_cal(dsg_cal(data_tensor))
        probs_cal = F.softmax(logits_cal, dim=-1)
        topk_p_cal, topk_i_cal = torch.topk(probs_cal, k=TOP_K, dim=-1)
        mask_cal = torch.zeros_like(probs_cal).scatter_(-1, topk_i_cal, 1.0)
        
        with torch.no_grad():
            cal_tokens += mask_cal.sum(dim=(0, 1))

        expert_outs_cal = torch.stack([exp(data_tensor) for exp in experts_cal], dim=2)
        out_cal_raw = ScaledUnGatedFunction.apply(
            expert_outs_cal, probs_cal.unsqueeze(-1), mask_cal.unsqueeze(-1), 1.0
        )
        out_cal = out_cal_raw.sum(dim=2)
        
        tokens_per_exp_c = mask_cal.sum(dim=(0, 1)) / total_tokens_per_step
        aux_loss_c = NUM_EXPERTS * torch.sum(tokens_per_exp_c * probs_cal.mean(dim=(0, 1)))
        loss_cal = F.mse_loss(out_cal, data_tensor) + (0.001 * torch.mean(torch.logsumexp(logits_cal, dim=-1)**2)) + (0.15 * aux_loss_c)
        loss_cal.backward()
        opt_cal.step()

    std_dead_heads = (std_tokens == 0).sum().item()
    cal_dead_heads = (cal_tokens == 0).sum().item()

    print(f"\n" + "="*85)
    print(f"RESULTS FOR: {domain_name}")
    print("="*85)
    print(f"  Standard MoE Final MSE Loss      : {loss_std.item():.6f} | Dead Experts: {std_dead_heads}/{NUM_EXPERTS}")
    print(f"  Calibrated MoE Final MSE Loss    : {loss_cal.item():.6f} | Dead Experts: {cal_dead_heads}/{NUM_EXPERTS}")
    print("-" * 85)

print("="*95)
