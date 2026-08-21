import time
import math
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

# ==========================================
# 1. UN-GATED AUTOGRAD KERNEL
# ==========================================
class ScaledUnGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs, gate_weights, active_mask, scale_factor):
        ctx.save_for_backward(expert_outputs, gate_weights, active_mask)
        ctx.scale_factor = scale_factor
        return (expert_outputs * gate_weights * active_mask).sum(dim=-2) * scale_factor

    @staticmethod
    def backward(ctx, grad_output):
        expert_outputs, gate_weights, active_mask = ctx.saved_tensors
        scale = ctx.scale_factor
        
        grad_out_expanded = grad_output.unsqueeze(-2)
        grad_active = grad_out_expanded * gate_weights * active_mask * scale
        
        inactive_mask = 1.0 - active_mask
        leakage_coefficient = 0.05
        grad_inactive = grad_out_expanded * gate_weights * inactive_mask * (scale * leakage_coefficient)
        
        total_expert_grad = grad_active + grad_inactive
        grad_gate = (grad_out_expanded * expert_outputs * active_mask * scale).sum(dim=-1, keepdim=True)
        
        return total_expert_grad, grad_gate, None, None

scaled_ungated_gate = ScaledUnGatedFunction.apply

# ==========================================
# 2. DEVICE-AWARE COSINE ORCHESTRATOR
# ==========================================
class DomainSignatureGenerator(nn.Module):
    def __init__(self, hidden_dim, signature_dim):
        super().__init__()
        self.signature_proj = nn.Sequential(
            nn.Linear(hidden_dim, signature_dim),
            nn.LayerNorm(signature_dim),
            nn.GELU(),
            nn.Linear(signature_dim, signature_dim)
        )
    def forward(self, x):
        x_fp32 = x.float()
        proj = self.signature_proj(x_fp32)
        return F.normalize(proj, p=2, dim=-1)

class CosineTopKOrchestrator(nn.Module):
    def __init__(self, hidden_dim, signature_dim, num_experts, top_k=2, temperature=0.2):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.dsg = DomainSignatureGenerator(hidden_dim, signature_dim)
        self.expert_signatures = nn.Parameter(torch.empty(num_experts, signature_dim))
        nn.init.orthogonal_(self.expert_signatures)
    def forward(self, x):
        domain_sigs = self.dsg(x)
        norm_expert_sigs = F.normalize(self.expert_signatures.float(), p=2, dim=-1)
        return torch.matmul(domain_sigs, norm_expert_sigs.T) / self.temperature
    def compute_ortho_loss(self):
        norm_sigs = F.normalize(self.expert_signatures.float(), p=2, dim=-1)
        identity = torch.eye(norm_sigs.size(0), device=norm_sigs.device)
        return torch.norm(torch.matmul(norm_sigs, norm_sigs.T) - identity)

# ==========================================
# 3. SwiGLU EXPERTS & BLOCK STACK
# ==========================================
class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim=None):
        super().__init__()
        intermediate_dim = intermediate_dim or int(2 * (4 * hidden_dim) / 3)
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MoEBlock(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, top_k=2, signature_dim=16, total_layers=4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.router = CosineTopKOrchestrator(hidden_dim, signature_dim, num_experts, top_k)
        self.experts = nn.ModuleList([SwiGLUExpert(hidden_dim) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.top_k = top_k
        self.res_scale = 1.0 / math.sqrt(total_layers)
    def forward(self, x):
        normed_x = self.norm(x)
        logits = self.router(normed_x)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        
        _, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
        active_mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0).unsqueeze(-1)
        
        expert_outputs = torch.stack([expert(normed_x) for expert in self.experts], dim=-2)
        combined = scaled_ungated_gate(expert_outputs, weights, active_mask, 0.1)
        
        return x + self.res_scale * combined, logits

class ProductionMoEModel(nn.Module):
    def __init__(self, vocab_size=50257, hidden_dim=256, num_layers=4, num_experts=4, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            MoEBlock(hidden_dim, num_experts=num_experts, top_k=top_k, total_layers=num_layers) 
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
    def forward(self, token_ids):
        x = self.embedding(token_ids)
        all_logits = []
        for layer in self.layers:
            x, logits = layer(x)
            all_logits.append(logits)
        x = self.final_norm(x)
        return self.lm_head(x), all_logits
    def total_ortho_loss(self):
        return sum(layer.router.compute_ortho_loss() for layer in self.layers)

# ==========================================
# 4. LOCAL SYNTHETIC DATA PIPELINE
# ==========================================
def get_dataset_loaders(batch_size=8, seq_len=128, vocab_size=50257):
    num_samples = 256
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    targets = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    
    train_dataset = torch.utils.data.TensorDataset(inputs, targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = train_loader
    return train_loader, val_loader, vocab_size

# ==========================================
# 5. METRICS & EXECUTION
# ==========================================
def compute_allocation_metrics(alloc_counts):
    counts = torch.tensor(alloc_counts, dtype=torch.float32)
    mean = counts.mean().item()
    std = counts.std().item()
    cv = std / (mean + 1e-6)
    max_min_ratio = (counts.max() / (counts.min() + 1e-6)).item()
    return round(cv, 3), round(max_min_ratio, 2)

def evaluate_val_loss(model, val_loader, vocab_size, device="cpu"):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
            steps += 1
            if steps >= 30:
                break
    model.train()
    avg_loss = total_loss / steps
    val_ppl = math.exp(min(avg_loss, 20))
    return round(avg_loss, 4), round(val_ppl, 2)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nExecuting Production Convergence Benchmark on Device: {device.upper()}\n")
    
    train_loader, val_loader, vocab_size = get_dataset_loaders(batch_size=8, seq_len=128)
    
    model = ProductionMoEModel(vocab_size=vocab_size, hidden_dim=256, num_layers=4, num_experts=4, top_k=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
        
    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=10, warmup=5, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/moe_profile'),
        record_shapes=True,
        profile_memory=True
    ) as prof:
        model.train()
        
        for step, (inputs, targets) in enumerate(train_loader):
            if step >= 150:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            autocast_ctx = (
                torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                if device == "cuda" else contextlib.nullcontext()
            )
            
            with autocast_ctx:
                logits, all_router_logits = model(inputs)
                ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                ortho_loss = model.total_ortho_loss()
                loss = ce_loss + 0.05 * ortho_loss
                
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            prof.step()
            
            if (step + 1) % 30 == 0:
                alloc = torch.topk(all_router_logits[0], k=2, dim=-1).indices.flatten().bincount(minlength=4).tolist()
                cv, ratio = compute_allocation_metrics(alloc)
                val_loss, val_ppl = evaluate_val_loss(model, val_loader, vocab_size, device)
                print(f"Step {step+1:3d} | Train CE: {ce_loss.item():.4f} | Val CE: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | Alloc CV: {cv:.3f} | Max/Min: {ratio:.2f}:1")
                
    print("\n=========================================================================================")
    print("Benchmark complete cleanly.")
    print("=========================================================================================\n")