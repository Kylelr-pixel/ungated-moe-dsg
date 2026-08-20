import torch
import time
from my_code import ScaledUnGatedFunction, DomainSignatureGenerator

# Initialize test dimensions
batch_size = 32
seq_len = 128
hidden_dim = 256
signature_dim = 64
num_experts = 8
num_steps = 1000

print(f"Starting extended benchmark across {num_steps} iterations...")
dsg = DomainSignatureGenerator(hidden_dim, signature_dim)

# Linear projection to map signature dimension (64) to num_experts (8)
expert_proj = torch.nn.Linear(signature_dim, num_experts)

# Define expert weights matching (num_experts, hidden_dim)
expert_weights = torch.randn(num_experts, hidden_dim, requires_grad=True)

start_time = time.time()
for step in range(1, num_steps + 1):
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 1. Forward Pass through DSG (outputs [32, 128, 64])
    sig_weights = dsg(x)
    
    # 2. Project signature_dim (64) -> num_experts (8) -> shape: [32, 128, 8]
    gate_logits = expert_proj(sig_weights)
    gate_weights = torch.sigmoid(gate_logits)
    active_mask = (gate_weights > 0.5).float()
    
    # 3. Compute expert outputs: [32, 128, 8, 256]
    # x: [32, 128, 256] -> unsqueeze to [32, 128, 1, 256]
    # expert_weights: [8, 256]
    expert_outputs = x.unsqueeze(2) * expert_weights
    
    # 4. Custom Autograd Kernel Execution
    out = ScaledUnGatedFunction.apply(
        expert_outputs, 
        gate_weights.unsqueeze(-1), 
        active_mask.unsqueeze(-1), 
        0.1
    )
    
    loss = out.pow(2).mean()
    
    # 5. Backward Pass
    loss.backward()
    
    if step % 200 == 0 or step == 1:
        print(f"Step [{step}/{num_steps}] - Loss: {loss.item():.6f}")

elapsed = time.time() - start_time
print(f"Benchmark Complete in {elapsed:.2f} seconds!")
