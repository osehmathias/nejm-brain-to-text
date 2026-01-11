"""
Debug script to compare how CTC loss gradients flow through RNN vs Conformer.
This tests if the output scale difference affects learning dynamics.
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/ubuntu/nejm/nejm-brain-to-text-repo/model_training')

from rnn_model import GRUDecoder
from conformer_model import ConformerDecoder

# Params
neural_dim = 512
n_days = 45
n_classes = 41
batch_size = 8
seq_len = 200

print("=" * 60)
print("TEST 1: CTC Loss at Initialization")
print("=" * 60)

# Create models
rnn = GRUDecoder(
    neural_dim=neural_dim, n_units=768, n_days=n_days, n_classes=n_classes,
    rnn_dropout=0.0, input_dropout=0.0, n_layers=5, patch_size=14, patch_stride=4,
)

conformer = ConformerDecoder(
    neural_dim=neural_dim, n_days=n_days, n_classes=n_classes,
    d_model=768, n_heads=8, d_ff=3072, n_layers=6, conv_kernel_size=31,
    dropout=0.0, input_dropout=0.0, patch_size=14, patch_stride=4,
    gradient_checkpointing=False,
)

# Create fake batch
torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, neural_dim)
day_idx = torch.randint(0, n_days, (batch_size,))

# Fake targets (random phoneme sequences)
target_lengths = torch.randint(10, 30, (batch_size,))
targets = torch.randint(1, n_classes, (target_lengths.sum().item(),))  # Avoid blank (0)

# Forward pass
rnn.train()
conformer.train()

rnn_out = rnn(x, day_idx)
conf_out = conformer(x, day_idx)

# CTC loss expects (T, N, C) format
rnn_out_ctc = rnn_out.permute(1, 0, 2).log_softmax(dim=-1)
conf_out_ctc = conf_out.permute(1, 0, 2).log_softmax(dim=-1)

input_lengths = torch.full((batch_size,), rnn_out.size(1), dtype=torch.long)

ctc_loss = nn.CTCLoss(blank=0, reduction='mean')

rnn_loss = ctc_loss(rnn_out_ctc, targets, input_lengths, target_lengths)
conf_loss = ctc_loss(conf_out_ctc, targets, input_lengths, target_lengths)

print(f"RNN CTC loss at init: {rnn_loss.item():.4f}")
print(f"Conformer CTC loss at init: {conf_loss.item():.4f}")
print(f"Ratio: {conf_loss.item() / rnn_loss.item():.2f}x")

print("\n" + "=" * 60)
print("TEST 2: Gradient Magnitudes")
print("=" * 60)

# Backward pass
rnn.zero_grad()
conformer.zero_grad()

rnn_loss.backward()
conf_loss.backward()

# Check gradients on output layer
rnn_out_grad = rnn.out.weight.grad
conf_out_grad = conformer.output.weight.grad

print(f"RNN output layer grad - mean: {rnn_out_grad.abs().mean():.6f}, max: {rnn_out_grad.abs().max():.6f}")
print(f"Conformer output layer grad - mean: {conf_out_grad.abs().mean():.6f}, max: {conf_out_grad.abs().max():.6f}")
print(f"Ratio: {conf_out_grad.abs().mean() / rnn_out_grad.abs().mean():.2f}x")

# Check gradients on day layers
rnn_day_grad = torch.stack([rnn.day_weights[i].grad for i in range(n_days) if rnn.day_weights[i].grad is not None])
conf_day_grad = conformer.day_weights.grad

print(f"\nRNN day layer grad - mean: {rnn_day_grad.abs().mean():.6f}")
print(f"Conformer day layer grad - mean: {conf_day_grad.abs().mean():.6f}")
print(f"Ratio: {conf_day_grad.abs().mean() / rnn_day_grad.abs().mean():.2f}x")

print("\n" + "=" * 60)
print("TEST 3: Prediction Distribution at Init")
print("=" * 60)

with torch.no_grad():
    rnn_preds = rnn_out.argmax(dim=-1)
    conf_preds = conf_out.argmax(dim=-1)

    # Count how often each class is predicted
    rnn_class_counts = torch.bincount(rnn_preds.flatten(), minlength=n_classes)
    conf_class_counts = torch.bincount(conf_preds.flatten(), minlength=n_classes)

    print(f"RNN predictions - unique classes: {(rnn_class_counts > 0).sum().item()}/{n_classes}")
    print(f"Conformer predictions - unique classes: {(conf_class_counts > 0).sum().item()}/{n_classes}")

    # Most predicted class frequency
    print(f"\nRNN - most common class: {rnn_class_counts.argmax().item()} ({rnn_class_counts.max().item()}/{rnn_preds.numel()} = {rnn_class_counts.max().item()/rnn_preds.numel()*100:.1f}%)")
    print(f"Conformer - most common class: {conf_class_counts.argmax().item()} ({conf_class_counts.max().item()}/{conf_preds.numel()} = {conf_class_counts.max().item()/conf_preds.numel()*100:.1f}%)")

print("\n" + "=" * 60)
print("TEST 4: What if we scale Conformer output?")
print("=" * 60)

# Scale conformer output to match RNN scale
scale_factor = 0.0686 / 1.4062  # RNN std / Conformer std

with torch.no_grad():
    # Manually scale the output layer weights
    conformer.output.weight.data *= scale_factor
    conformer.output.bias.data *= scale_factor

# Forward again
conf_out_scaled = conformer(x, day_idx)
conf_out_scaled_ctc = conf_out_scaled.permute(1, 0, 2).log_softmax(dim=-1)
conf_loss_scaled = ctc_loss(conf_out_scaled_ctc, targets, input_lengths, target_lengths)

print(f"Conformer logits after scaling - mean: {conf_out_scaled.mean():.4f}, std: {conf_out_scaled.std():.4f}")
print(f"Conformer CTC loss after scaling: {conf_loss_scaled.item():.4f}")
print(f"RNN CTC loss (reference): {rnn_loss.item():.4f}")

# Entropy check
conf_probs_scaled = torch.softmax(conf_out_scaled, dim=-1)
entropy_scaled = -(conf_probs_scaled * conf_probs_scaled.log()).sum(-1).mean()
print(f"Conformer entropy after scaling: {entropy_scaled:.4f}")
print(f"Max entropy: {torch.log(torch.tensor(n_classes, dtype=torch.float)):.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
If scaling the Conformer output brings:
  1. CTC loss close to RNN's loss
  2. Entropy close to max entropy
  3. Gradient magnitudes closer to RNN's

Then output scale IS the problem, and we should fix the initialization.

If they're still very different, there are other issues at play.
""")
