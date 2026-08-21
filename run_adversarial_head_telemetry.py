import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from my_code import ScaledUnGatedFunction, SwiGLUExpert

# --- PER-HEAD AUDITED ADVERSARIAL STRESS TEST ---
BATCH_SIZE = 8
SEQ_LEN = 32
VOCAB_SIZE = 1000
EMBED_DIM = 128
INTERMEDIATE_DIM = 256
SIGNATURE_DIM = 32
NUM_EXPERTS = 16
TOP_K = 2
STEPS_PER_PHASE = 10

print("="*95)
print(f"  PER-HEAD ADVERSARIAL STRESS & CALIBRATION TELEMETRY (Top-{TOP_K} Sparse)")
print("="*95)

token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
dsg = nn.Sequential(nn.Linear(EMBED_DIM, SIGNATURE_DIM), nn.ReLU())
proj = nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
experts = nn.ModuleList([SwiGLUExpert(EMBED_DIM, INTERMEDIATE_DIM) for _ in range(NUM_EXPERTS)])

optimizer = torch.optim.AdamW(
    list(token_embedding.parameters()) + list(dsg.parameters()) + list(proj.parameters()) + list(experts.parameters()), 
    lr=1e-3
)

phases = [
    ("Phase 1: Baseline Clean Text Stream", lambda: torch.randint(10, 500, (BATCH_SIZE, SEQ_LEN))),
    ("Phase 2: Adversarial High-Entropy Poisoning (Gibberish)", lambda: torch.randint(900, 999, (BATCH_SIZE, SEQ_LEN))),
    ("Phase 3: Extreme Sequence Length Burst (4x Token Load)", lambda: torch.randint(10, 500, (BATCH_SIZE, SEQ_LEN * 4))),
    ("Phase 4: Targeted Expert Starvation Attack", lambda: torch.full((BATCH_SIZE, SEQ_LEN), 42, dtype=torch.long))
]

total_tokens_processed = 0
start_time = time.time()
total_tokens_per_step = BATCH_SIZE * SEQ_LEN * TOP_K

for phase_name, batch_gen in phases:
    print(f"\n" + "="*95)
    print(f"INJECTING: {phase_name}")
    print("="*95)
    print(f"{'Step':<6} | {'MSE Loss':<12} | {'Active Tokens':<15} | {'Router Confidence':<18} | {'Calibration Trigger Status'}")
    print("-" * 95)
    
    # Track cumulative metrics per head across this phase
    phase_head_tokens = torch.zeros(NUM_EXPERTS)
    phase_active_grads = torch.zeros(NUM_EXPERTS)
    phase_inactive_grads = torch.zeros(NUM_EXPERTS)
    
    for step in range(1, STEPS_PER_PHASE + 1):
        optimizer.zero_grad()
        token_ids = batch_gen()
        b_s, s_l = token_ids.shape
        curr_tokens = b_s * s_l
        
        embedded_input = token_embedding(token_ids)
        sig_weights = dsg(embedded_input.mean(dim=1))
        logits = proj(sig_weights)
        probs = F.softmax(logits, dim=-1)
        
        router_confidence = probs.max(dim=-1).values.mean().item()
        
        topk_p, topk_i = torch.topk(probs, k=TOP_K, dim=-1)
        mask = torch.zeros_like(probs).scatter_(-1, topk_i, 1.0)
        
        with torch.no_grad():
            phase_head_tokens += mask.sum(dim=0)
            
        seq_mask = mask.unsqueeze(1).expand(-1, s_l, -1).unsqueeze(-1)
        seq_probs = probs.unsqueeze(1).expand(-1, s_l, -1).unsqueeze(-1)
        
        expert_outs = torch.stack([exp(embedded_input) for exp in experts], dim=2)
        
        out_raw = ScaledUnGatedFunction.apply(
            expert_outs, seq_probs, seq_mask, 1.0
        )
        out = out_raw.sum(dim=2)
        
        tokens_per_exp = mask.sum(dim=0) / (b_s * s_l * TOP_K)
        aux_loss = NUM_EXPERTS * torch.sum(tokens_per_exp * probs.mean(dim=0))
        loss = F.mse_loss(out, embedded_input) + (0.001 * torch.mean(torch.logsumexp(logits, dim=-1)**2)) + (0.15 * aux_loss)
        
        loss.backward()
        
        step_tokens = mask.sum(dim=0)
        calibration_triggered = False
        with torch.no_grad():
            for i, expert in enumerate(experts):
                if expert.w1.weight.grad is not None:
                    g_norm = expert.w1.weight.grad.norm().item()
                    if step_tokens[i].item() > 0:
                        phase_active_grads[i] += g_norm
                    else:
                        phase_inactive_grads[i] += g_norm
                        if g_norm > 0:
                            calibration_triggered = True
                            
        optimizer.step()
        total_tokens_processed += curr_tokens
        
        calib_status = "ACTIVE (Inactive Grads Flowing)" if calibration_triggered else "Idle/None"
        print(f"Step {step:<2} | {loss.item():.6f}     | {curr_tokens:<15} | {router_confidence:<18.4f} | {calib_status}")

    print("\n" + "-"*85)
    print(f"PER-HEAD TELEMETRY SUMMARY FOR: {phase_name}")
    print("-" * 85)
    print(f"  {'Head ID':<8} | {'Tokens Routed':<16} | {'Avg Active Grad':<18} | {'Avg Inactive Grad (Calibration)'}")
    print("  " + "-" * 75)
    
    for i in range(NUM_EXPERTS):
        cnt = int(phase_head_tokens[i].item())
        act_g = phase_active_grads[i].item() / STEPS_PER_PHASE
        inact_g = phase_inactive_grads[i].item() / STEPS_PER_PHASE
        print(f"  Head {i:<2}   | {cnt:<16} | {act_g:<18.6f} | {inact_g:<18.6f}")

elapsed = time.time() - start_time
print("="*95)
print(f" ADVERSARIAL TELEMETRY AUDIT COMPLETE:")
print(f"   - Total Processed Tokens        : {total_tokens_processed:,} tokens")
print(f"   - Execution Time                : {elapsed:.4f} seconds")
print("="*95)
