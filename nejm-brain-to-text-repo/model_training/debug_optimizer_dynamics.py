"""
Debug script to analyze optimizer dynamics with different hyperparameters.
Covers checklist items: 2.1 (LR), 2.5 (epsilon), 2.6 (weight decay)

Key insight: RNN uses epsilon=0.1 (very large) while Conformer uses epsilon=1e-6.
This dramatically affects early training dynamics.
"""
import torch
import torch.nn as nn
import sys
import math
sys.path.insert(0, '/Users/oseh/code/river/research/nejm-brain-to-text/nejm-brain-to-text-repo/model_training')

from conformer_model import ConformerDecoder

# Params
neural_dim = 512
n_days = 45
n_classes = 41
batch_size = 8
seq_len = 200

print("=" * 70)
print("OPTIMIZER DYNAMICS ANALYSIS")
print("=" * 70)
print("""
Adam update formula: param -= lr * m / (sqrt(v) + epsilon)

Where:
  m = beta1 * m + (1-beta1) * grad        (momentum)
  v = beta2 * v + (1-beta2) * grad^2      (variance)

At step 1 (before bias correction):
  m ≈ (1-beta1) * grad = 0.1 * grad      (for beta1=0.9)
  v ≈ (1-beta2) * grad^2 = 0.02 * grad^2 (for beta2=0.98)

Effective update ≈ lr * 0.1 * grad / (sqrt(0.02) * |grad| + epsilon)
                 ≈ lr * 0.1 / (0.14 + epsilon/|grad|)

With epsilon=0.1 and |grad|=0.001: denominator ≈ 0.14 + 100 = 100.14 → dampened
With epsilon=1e-6 and |grad|=0.001: denominator ≈ 0.14 + 0.001 = 0.141 → NOT dampened
""")

# Create model
torch.manual_seed(42)
model = ConformerDecoder(
    neural_dim=neural_dim, n_days=n_days, n_classes=n_classes,
    d_model=768, n_heads=8, d_ff=3072, n_layers=6, conv_kernel_size=31,
    dropout=0.0, input_dropout=0.0, patch_size=14, patch_stride=4,
    gradient_checkpointing=False,
)
model.train()

# Get a gradient
torch.manual_seed(123)
x = torch.randn(batch_size, seq_len, neural_dim)
day_idx = torch.randint(0, n_days, (batch_size,))
targets = torch.randint(1, n_classes, (batch_size * 15,))
target_lengths = torch.full((batch_size,), 15, dtype=torch.long)

out = model(x, day_idx)
log_probs = out.permute(1, 0, 2).log_softmax(dim=-1)
input_lengths = torch.full((batch_size,), out.size(1), dtype=torch.long)

ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
loss.backward()

# Analyze gradient statistics
print("=" * 70)
print("GRADIENT STATISTICS AT INITIALIZATION")
print("=" * 70)

grad_stats = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        g = param.grad
        grad_stats[name] = {
            'mean': g.mean().item(),
            'std': g.std().item(),
            'abs_mean': g.abs().mean().item(),
            'max': g.abs().max().item(),
        }

# Print summary by component
print(f"\n{'Component':<40} {'Grad Mean':>12} {'Grad Std':>12} {'Grad |Max|':>12}")
print("-" * 80)

components = {
    'day_weights': [],
    'day_biases': [],
    'input_projection': [],
    'conformer': [],
    'output': [],
}

for name, stats in grad_stats.items():
    if 'day_weight' in name:
        components['day_weights'].append(stats)
    elif 'day_bias' in name:
        components['day_biases'].append(stats)
    elif 'input_projection' in name:
        components['input_projection'].append(stats)
    elif 'output' in name:
        components['output'].append(stats)
    else:
        components['conformer'].append(stats)

for comp_name, stats_list in components.items():
    if stats_list:
        avg_abs_mean = sum(s['abs_mean'] for s in stats_list) / len(stats_list)
        avg_std = sum(s['std'] for s in stats_list) / len(stats_list)
        max_val = max(s['max'] for s in stats_list)
        print(f"{comp_name:<40} {avg_abs_mean:>12.6f} {avg_std:>12.6f} {max_val:>12.6f}")

print("\n" + "=" * 70)
print("SIMULATED ADAM UPDATES (First Step)")
print("=" * 70)

# Test different epsilon values
epsilons = [1e-8, 1e-6, 1e-4, 1e-2, 0.1]
lrs = [1e-4, 3e-4, 1e-3]

# Pick output layer as representative
output_grad = model.output.weight.grad.clone()
output_weight = model.output.weight.data.clone()

print(f"\nOutput layer weight: mean={output_weight.mean():.6f}, std={output_weight.std():.6f}")
print(f"Output layer grad: mean={output_grad.abs().mean():.6f}, max={output_grad.abs().max():.6f}")

print(f"\n{'LR':<10} {'Epsilon':<12} {'Update Std':>14} {'Update/Weight %':>16} {'Status':<20}")
print("-" * 75)

for lr in lrs:
    for eps in epsilons:
        # Simulate first Adam step (simplified, ignoring bias correction for illustration)
        beta1, beta2 = 0.9, 0.98
        m = (1 - beta1) * output_grad
        v = (1 - beta2) * output_grad ** 2

        # Bias correction
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)

        update = lr * m_hat / (torch.sqrt(v_hat) + eps)

        update_std = update.std().item()
        weight_std = output_weight.std().item()
        pct = (update_std / weight_std) * 100

        status = ""
        if pct > 10:
            status = "TOO LARGE"
        elif pct > 1:
            status = "aggressive"
        elif pct < 0.01:
            status = "too small"
        else:
            status = "reasonable"

        print(f"{lr:<10.0e} {eps:<12.0e} {update_std:>14.6f} {pct:>15.2f}% {status:<20}")

print("\n" + "=" * 70)
print("WARMUP ANALYSIS")
print("=" * 70)

# Current config: lr_max=0.001, warmup=2000
# After warmup, full LR kicks in

warmup_schedules = [
    ("Current (2000 steps)", 2000, 0.001),
    ("Longer warmup (4000 steps)", 4000, 0.001),
    ("Longer warmup (8000 steps)", 8000, 0.001),
    ("Lower LR + standard warmup", 2000, 0.0003),
    ("Lower LR + longer warmup", 4000, 0.0003),
]

print(f"\n{'Schedule':<35} {'LR at step 1000':>18} {'LR at step 2000':>18} {'LR at step 4000':>18}")
print("-" * 95)

for name, warmup_steps, lr_max in warmup_schedules:
    def get_lr(step):
        if step < warmup_steps:
            return lr_max * step / warmup_steps
        return lr_max

    lr_1000 = get_lr(1000)
    lr_2000 = get_lr(2000)
    lr_4000 = get_lr(4000)

    print(f"{name:<35} {lr_1000:>18.6f} {lr_2000:>18.6f} {lr_4000:>18.6f}")

print("\n" + "=" * 70)
print("WEIGHT DECAY IMPACT")
print("=" * 70)

# RNN: weight_decay=0.001, Conformer: weight_decay=0.01
weight_decays = [0.001, 0.01, 0.1]

print(f"\n{'Weight Decay':<15} {'Decay per 1000 steps':>25} {'After 10k steps':>20}")
print("-" * 65)

for wd in weight_decays:
    # Weight decay in AdamW: w = w - lr * wd * w
    # Effective decay factor per step: (1 - lr * wd)
    # After n steps: w * (1 - lr * wd)^n

    lr = 0.001  # max lr
    decay_1000 = (1 - lr * wd) ** 1000
    decay_10000 = (1 - lr * wd) ** 10000

    print(f"{wd:<15.4f} {decay_1000:>25.4f} {decay_10000:>20.4f}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("""
Based on this analysis:

1. EPSILON: The RNN uses epsilon=0.1 which heavily dampens updates when gradients
   are small. This provides implicit stability. Consider increasing Conformer's
   epsilon from 1e-6 to at least 1e-4, or even 0.01-0.1 for better stability.

2. LEARNING RATE: With epsilon=1e-6, lr=0.001 may be too aggressive.
   Consider lr=3e-4 or lower.

3. WARMUP: 2000 steps may not be enough. By step 2000, full LR kicks in
   but the model may not have learned good representations yet.
   Consider 4000-8000 warmup steps.

4. WEIGHT DECAY: 0.01 weight decay with lr=0.001 causes ~10% weight shrinkage
   per 10k steps. This is aggressive - may cause underfitting early in training.
   Consider reducing to 0.001 to match RNN.

SUGGESTED EXPERIMENT ORDER:
1. Increase epsilon to 1e-4 or 0.01 (fastest to test)
2. Lower LR to 3e-4
3. Increase warmup to 4000
4. If still unstable, try epsilon=0.1 (matching RNN)
""")
