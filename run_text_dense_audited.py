import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from my_code import SwiGLUExpert

# --- AUDITED 8-DOMAIN DENSE TEXT BENCHMARK (WITH TOKEN & TENSOR LOSS TRACKING) ---
BATCH_SIZE = 8
SEQ_LEN = 32
VOCAB_SIZE = 1000
EMBED_DIM = 128
INTERMEDIATE_DIM = 256
SIGNATURE_DIM = 32
NUM_EXPERTS = 16
STEPS_PER_DOMAIN = 10

print("="*95)
print(f"  AUDITED 8-DOMAIN DENSE TEXT BENCHMARK (Token Loss & Shape Verification Active)")
print("="*95)

def get_text_batch(domain_type):
    torch.manual_seed(hash(domain_type) % 10000)
    if "Source Code" in domain_type: return torch.randint(10, 120, (BATCH_SIZE, SEQ_LEN))
    elif "Medical" in domain_type: return torch.randint(120, 240, (BATCH_SIZE, SEQ_LEN))
    elif "Legal" in domain_type: return torch.randint(240, 360, (BATCH_SIZE, SEQ_LEN))
    elif "Conversational" in domain_type: return torch.randint(360, 480, (BATCH_SIZE, SEQ_LEN))
    elif "Financial" in domain_type: return torch.randint(480, 600, (BATCH_SIZE, SEQ_LEN))
    elif "Historical" in domain_type: return torch.randint(600, 720, (BATCH_SIZE, SEQ_LEN))
    elif "Scientific" in domain_type: return torch.randint(720, 840, (BATCH_SIZE, SEQ_LEN))
    else: return torch.randint(840, 990, (BATCH_SIZE, SEQ_LEN))

domains = [
    "Domain 1: Source Code Syntax",
    "Domain 2: Medical Clinical Prose",
    "Domain 3: Legal Regulatory Clauses",
    "Domain 4: Conversational Narrative",
    "Domain 5: Financial Market Reports",
    "Domain 6: Historical Archives",
    "Domain 7: Scientific Research Abstract",
    "Domain 8: Technical Documentation"
]

token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
dsg = nn.Sequential(nn.Linear(EMBED_DIM, SIGNATURE_DIM), nn.ReLU())
proj = nn.Linear(SIGNATURE_DIM, NUM_EXPERTS)
experts = nn.ModuleList([SwiGLUExpert(EMBED_DIM, INTERMEDIATE_DIM) for _ in range(NUM_EXPERTS)])

optimizer = torch.optim.AdamW(
    list(token_embedding.parameters()) + list(dsg.parameters()) + list(proj.parameters()) + list(experts.parameters()), 
    lr=1e-3
)

total_system_tokens_processed = 0
start_time_global = time.time()

for domain_name in domains:
    token_ids = get_text_batch(domain_name)
    
    # Audit token batch dimensions
    b_size, s_len = token_ids.shape
    expected_tokens = b_size * s_len
    
    head_token_counts = torch.zeros(NUM_EXPERTS)
    head_grad_norms = torch.zeros(NUM_EXPERTS)
    
    print(f"\n" + "="*85)
    print(f"STREAMING DOMAIN: {domain_name} [Expected Tokens/Step: {expected_tokens}]")
    print("="*85)
    print(f"{'Step':<6} | {'MSE Loss':<12} | {'Tokens Processed':<18} | {'Token Loss / Truncation Status'}")
    print("-" * 75)
    
    for step in range(1, STEPS_PER_DOMAIN + 1):
        step_start = time.time()
        optimizer.zero_grad()
        
        embedded_input = token_embedding(token_ids)
        
        # Verify no shape distortion or token loss in embedding layer
        assert embedded_input.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM), "Shape distortion detected!"
        
        sig_weights = dsg(embedded_input.mean(dim=1))
        logits = proj(sig_weights)
        probs = F.softmax(logits, dim=-1)
        
        expert_outs = torch.stack([exp(embedded_input) for exp in experts], dim=2)
        out = (expert_outs * probs.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        
        # Verify output matches input tensor shape (zero sequence shrinkage/token loss)
        assert out.shape == embedded_input.shape, "Output tensor size mismatch!"
        
        loss = F.mse_loss(out, embedded_input)
        loss.backward()
        
        with torch.no_grad():
            head_token_counts += probs.sum(dim=0)
            for i, expert in enumerate(experts):
                if expert.w1.weight.grad is not None:
                    head_grad_norms[i] += expert.w1.weight.grad.norm().item()
                    
        optimizer.step()
        
        total_system_tokens_processed += expected_tokens
        print(f"Step {step:<2} | {loss.item():.6f}     | {expected_tokens} tokens         | ZERO Loss (Full Batch Preserved)")

    print("\n" + "-"*75)
    print(f"AUDIT SUMMARY FOR: {domain_name}")
    print("-" * 75)
    print(f"  {'Head ID':<8} | {'Cumulative Routing Weight':<26} | {'Avg Grad Norm':<18}")
    print("  " + "-" * 65)
    
    for i in range(NUM_EXPERTS):
        weight_sum = head_token_counts[i].item()
        avg_g = head_grad_norms[i].item() / STEPS_PER_DOMAIN
        print(f"  Head {i:<2}   | {weight_sum:<26.4f} | {avg_g:<18.6f}")

elapsed_global = time.time() - start_time_global
print("="*95)
print(f" GLOBAL EXECUTION AUDIT COMPLETE:")
print(f"   - Total System Tokens Processed : {total_system_tokens_processed:,} tokens")
print(f"   - Total Execution Wall Time     : {elapsed_global:.4f} seconds")
print(f"   - Token Loss Rate               : 0.00% (Strict Shape Assertion Passed)")
print("="*95)
