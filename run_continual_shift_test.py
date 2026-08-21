import torch
import torch.nn as nn
import torch.nn.functional as F
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator, SwiGLUExpert

# --- GRANULAR HEAD-LEVEL CONTINUAL DOMAIN-SWITCHING TEST ---
BATCH_SIZE = 16
SEQ_LEN = 64
HIDDEN_DIM = 256
INTERMEDIATE_DIM = 512
SIGNATURE_DIM = 64
NUM_EXPERTS = 16
TOP_K = 2
STEPS_PER_SHIFT = 25

print("="*95)
print(f"  GRANULAR HEAD-LEVEL DOMAIN-SWITCHING BENCHMARK ({NUM_EXPERTS} Experts, Top-{TOP_K} Sparse)")
print("="*95)

torch.manual_seed(2026)
domains = [
    ("Domain A (High-Variance)", lambda: torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 4.0),
    ("Domain B (Sparse Spikes)", lambda: (torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) > 0.85).float() * 8.0),
    ("Domain C (Smooth Sine)", lambda: torch.sin(torch.linspace(0, 50, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM)).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 0.5),
    ("Domain D (Bimodal Out)", lambda: torch.bernoulli(torch.full((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), 0.5)) * 5.0 - 2.5),
    ("Domain E (High-Freq Osc)", lambda: torch.sin(torch.linspace(0, 200, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM)).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 2.0),
    ("Domain F (Smooth Cosine)", lambda: torch.cos(torch.linspace(0, 50, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM)).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 0.5),
    ("Domain G (Uniform Noise)", lambda: torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 6.0 - 3.0),
    ("Domain H (Step Function)", lambda: torch.sign(torch.sin(torch.linspace(0, 50, BATCH_SIZE * SEQ_LEN * HIDDEN_DIM))).view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) * 3.0)
]

dsg = DomainSignatureGenerator(HIDDEN_DIM, SIGNATURE_DIM)
proj = nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
experts = nn.ModuleList([SwiGLUExpert(HIDDEN_DIM, INTERMEDIATE_DIM) for _ in range(NUM_EXPERTS)])
optimizer = torch.optim.AdamW(list(dsg.parameters()) + list(proj.parameters()) + list(experts.parameters()), lr=1e-3)

total_tokens_per_step = BATCH_SIZE * SEQ_LEN * TOP_K

for domain_idx, (name, data_gen) in enumerate(domains):
    data_tensor = data_gen()
    head_token_counts = torch.zeros(NUM_EXPERTS)
    head_active_grads = torch.zeros(NUM_EXPERTS)
    head_inactive_grads = torch.zeros(NUM_EXPERTS)
    
    for step in range(1, STEPS_PER_SHIFT + 1):
        optimizer.zero_grad()
        
        logits = proj(dsg(data_tensor))
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = torch.topk(probs, k=TOP_K, dim=-1)
        mask = torch.zeros_like(probs).scatter_(-1, topk_i, 1.0)
        
        with torch.no_grad():
            head_token_counts += mask.sum(dim=(0, 1))

        expert_outs = torch.stack([exp(data_tensor) for exp in experts], dim=2)
        out_raw = ScaledUnGatedFunction.apply(
            expert_outs, probs.unsqueeze(-1), mask.unsqueeze(-1), 1.0
        )
        out = out_raw.sum(dim=2)
        
        tokens_per_exp = mask.sum(dim=(0, 1)) / total_tokens_per_step
        aux_loss = NUM_EXPERTS * torch.sum(tokens_per_exp * probs.mean(dim=(0, 1)))
        loss = F.mse_loss(out, data_tensor) + (0.001 * torch.mean(torch.logsumexp(logits, dim=-1)**2)) + (0.15 * aux_loss)
        
        loss.backward()
        
        step_token_counts = mask.sum(dim=(0, 1))
        with torch.no_grad():
            for i, expert in enumerate(experts):
                if expert.w1.weight.grad is not None:
                    g_norm = expert.w1.weight.grad.norm().item()
                    if step_token_counts[i].item() > 0:
                        head_active_grads[i] += g_norm
                    else:
                        head_inactive_grads[i] += g_norm
                        
        optimizer.step()

    print(f"\n" + "="*85)
    print(f"DOMAIN SHIFT BREAKDOWN: {name} (Loss: {loss.item():.6f})")
    print("="*85)
    print(f"  {'Head ID':<8} | {'Tokens Routed':<16} | {'Avg Active Grad':<18} | {'Avg Inactive Grad':<18}")
    print("  " + "-" * 75)
    
    total_domain_tokens = head_token_counts.sum().item()
    for i in range(NUM_EXPERTS):
        cnt = int(head_token_counts[i].item())
        pct = (cnt / total_domain_tokens) * 100 if total_domain_tokens > 0 else 0
        act_g = head_active_grads[i].item() / STEPS_PER_SHIFT
        inact_g = head_inactive_grads[i].item() / STEPS_PER_SHIFT
        print(f"  Head {i:<2}   | {cnt:<6} ({pct:5.1f}%) | {act_g:<18.6f} | {inact_g:<18.6f}")

print("="*95)
