"""
Debug script to verify gradient checkpointing produces identical results.
Covers checklist item: 1.3

Gradient checkpointing saves memory by not storing intermediate activations,
but recomputes them during backward pass. This should be numerically identical
(or very close) to standard training.
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/Users/oseh/code/river/research/nejm-brain-to-text/nejm-brain-to-text-repo/model_training')

from conformer_model import ConformerDecoder

# Params
neural_dim = 512
n_days = 45
n_classes = 41
batch_size = 8
seq_len = 200

print("=" * 70)
print("GRADIENT CHECKPOINTING VERIFICATION")
print("=" * 70)

# Create two identical models
torch.manual_seed(42)
model_no_ckpt = ConformerDecoder(
    neural_dim=neural_dim, n_days=n_days, n_classes=n_classes,
    d_model=768, n_heads=8, d_ff=3072, n_layers=6, conv_kernel_size=31,
    dropout=0.0, input_dropout=0.0, patch_size=14, patch_stride=4,
    gradient_checkpointing=False,
)

torch.manual_seed(42)
model_with_ckpt = ConformerDecoder(
    neural_dim=neural_dim, n_days=n_days, n_classes=n_classes,
    d_model=768, n_heads=8, d_ff=3072, n_layers=6, conv_kernel_size=31,
    dropout=0.0, input_dropout=0.0, patch_size=14, patch_stride=4,
    gradient_checkpointing=True,
)

# Verify weights are identical
print("\n[TEST 1] Weight initialization identical:")
weights_match = True
for (name1, p1), (name2, p2) in zip(model_no_ckpt.named_parameters(), model_with_ckpt.named_parameters()):
    if not torch.allclose(p1, p2):
        print(f"  MISMATCH: {name1}")
        weights_match = False
print(f"  All weights match: {weights_match}")

# Forward pass comparison
print("\n[TEST 2] Forward pass outputs:")
torch.manual_seed(123)
x = torch.randn(batch_size, seq_len, neural_dim)
day_idx = torch.randint(0, n_days, (batch_size,))

model_no_ckpt.train()
model_with_ckpt.train()

out_no_ckpt = model_no_ckpt(x.clone(), day_idx)
out_with_ckpt = model_with_ckpt(x.clone(), day_idx)

forward_match = torch.allclose(out_no_ckpt, out_with_ckpt, atol=1e-5)
print(f"  Outputs match (atol=1e-5): {forward_match}")
if not forward_match:
    diff = (out_no_ckpt - out_with_ckpt).abs()
    print(f"  Max diff: {diff.max():.2e}, Mean diff: {diff.mean():.2e}")

# Backward pass comparison
print("\n[TEST 3] Backward pass gradients:")

# Create fake CTC-like loss
targets = torch.randint(1, n_classes, (batch_size * 15,))
target_lengths = torch.full((batch_size,), 15, dtype=torch.long)
input_lengths = torch.full((batch_size,), out_no_ckpt.size(1), dtype=torch.long)

ctc_loss = nn.CTCLoss(blank=0, reduction='mean')

# Loss and backward - no checkpointing
model_no_ckpt.zero_grad()
out_no_ckpt = model_no_ckpt(x.clone(), day_idx)
log_probs_no_ckpt = out_no_ckpt.permute(1, 0, 2).log_softmax(dim=-1)
loss_no_ckpt = ctc_loss(log_probs_no_ckpt, targets, input_lengths, target_lengths)
loss_no_ckpt.backward()

# Loss and backward - with checkpointing
model_with_ckpt.zero_grad()
out_with_ckpt = model_with_ckpt(x.clone(), day_idx)
log_probs_with_ckpt = out_with_ckpt.permute(1, 0, 2).log_softmax(dim=-1)
loss_with_ckpt = ctc_loss(log_probs_with_ckpt, targets, input_lengths, target_lengths)
loss_with_ckpt.backward()

print(f"  Loss (no ckpt): {loss_no_ckpt.item():.6f}")
print(f"  Loss (with ckpt): {loss_with_ckpt.item():.6f}")
print(f"  Loss match: {torch.allclose(loss_no_ckpt, loss_with_ckpt, atol=1e-5)}")

# Compare gradients
print("\n[TEST 4] Gradient comparison by layer:")
grad_diffs = []
for (name1, p1), (name2, p2) in zip(model_no_ckpt.named_parameters(), model_with_ckpt.named_parameters()):
    if p1.grad is None or p2.grad is None:
        continue

    diff = (p1.grad - p2.grad).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    grad_diffs.append((name1, max_diff, mean_diff))

    if max_diff > 1e-4:
        print(f"  [LARGE DIFF] {name1}: max={max_diff:.2e}, mean={mean_diff:.2e}")

all_grads_match = all(d[1] < 1e-4 for d in grad_diffs)
print(f"\n  All gradients match (max_diff < 1e-4): {all_grads_match}")

if not all_grads_match:
    print("\n  Top 5 largest gradient differences:")
    sorted_diffs = sorted(grad_diffs, key=lambda x: x[1], reverse=True)[:5]
    for name, max_diff, mean_diff in sorted_diffs:
        print(f"    {name}: max={max_diff:.2e}, mean={mean_diff:.2e}")

# Test with multiple forward-backward passes (accumulation)
print("\n" + "=" * 70)
print("[TEST 5] Multiple forward-backward passes (gradient accumulation):")
print("=" * 70)

torch.manual_seed(42)
model_no_ckpt2 = ConformerDecoder(
    neural_dim=neural_dim, n_days=n_days, n_classes=n_classes,
    d_model=768, n_heads=8, d_ff=3072, n_layers=6, conv_kernel_size=31,
    dropout=0.0, input_dropout=0.0, patch_size=14, patch_stride=4,
    gradient_checkpointing=False,
)

torch.manual_seed(42)
model_with_ckpt2 = ConformerDecoder(
    neural_dim=neural_dim, n_days=n_days, n_classes=n_classes,
    d_model=768, n_heads=8, d_ff=3072, n_layers=6, conv_kernel_size=31,
    dropout=0.0, input_dropout=0.0, patch_size=14, patch_stride=4,
    gradient_checkpointing=True,
)

model_no_ckpt2.train()
model_with_ckpt2.train()
model_no_ckpt2.zero_grad()
model_with_ckpt2.zero_grad()

# Multiple mini-batches
n_accumulation_steps = 4
for step in range(n_accumulation_steps):
    torch.manual_seed(step * 100)
    x_step = torch.randn(batch_size, seq_len, neural_dim)
    day_idx_step = torch.randint(0, n_days, (batch_size,))
    targets_step = torch.randint(1, n_classes, (batch_size * 15,))

    # No checkpointing
    out1 = model_no_ckpt2(x_step.clone(), day_idx_step)
    log_probs1 = out1.permute(1, 0, 2).log_softmax(dim=-1)
    loss1 = ctc_loss(log_probs1, targets_step, input_lengths, target_lengths)
    (loss1 / n_accumulation_steps).backward()

    # With checkpointing
    out2 = model_with_ckpt2(x_step.clone(), day_idx_step)
    log_probs2 = out2.permute(1, 0, 2).log_softmax(dim=-1)
    loss2 = ctc_loss(log_probs2, targets_step, input_lengths, target_lengths)
    (loss2 / n_accumulation_steps).backward()

# Compare accumulated gradients
accumulated_grad_diffs = []
for (name1, p1), (name2, p2) in zip(model_no_ckpt2.named_parameters(), model_with_ckpt2.named_parameters()):
    if p1.grad is None or p2.grad is None:
        continue
    diff = (p1.grad - p2.grad).abs()
    accumulated_grad_diffs.append((name1, diff.max().item()))

all_accumulated_match = all(d[1] < 1e-4 for d in accumulated_grad_diffs)
print(f"  After {n_accumulation_steps} accumulation steps, all gradients match: {all_accumulated_match}")

if not all_accumulated_match:
    print("  Largest accumulated gradient differences:")
    sorted_acc = sorted(accumulated_grad_diffs, key=lambda x: x[1], reverse=True)[:5]
    for name, max_diff in sorted_acc:
        print(f"    {name}: {max_diff:.2e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

tests_passed = weights_match and forward_match and all_grads_match and all_accumulated_match

if tests_passed:
    print("ALL TESTS PASSED")
    print("Gradient checkpointing is working correctly.")
    print("It is NOT the cause of training instability.")
else:
    print("SOME TESTS FAILED")
    print("Gradient checkpointing may be causing numerical differences.")
    print("Consider training with gradient_checkpointing: false to verify.")
    print(f"  - Weights match: {weights_match}")
    print(f"  - Forward match: {forward_match}")
    print(f"  - Gradients match: {all_grads_match}")
    print(f"  - Accumulated grads match: {all_accumulated_match}")
