"""
Debug script to compare RNN and Conformer model outputs.
Checks if initial outputs are sensible or degenerate.
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/nejm/nejm-brain-to-text-repo/model_training')

from rnn_model import GRUDecoder
from conformer_model import ConformerDecoder

# Same params
neural_dim = 512
n_days = 45
n_classes = 41
batch_size = 8
seq_len = 200

print("=" * 60)
print("Creating models...")
print("=" * 60)

# Create models
rnn = GRUDecoder(
    neural_dim=neural_dim,
    n_units=768,
    n_days=n_days,
    n_classes=n_classes,
    rnn_dropout=0.4,
    input_dropout=0.2,
    n_layers=5,
    patch_size=14,
    patch_stride=4,
)

conformer = ConformerDecoder(
    neural_dim=neural_dim,
    n_days=n_days,
    n_classes=n_classes,
    d_model=768,
    n_heads=8,
    d_ff=3072,
    n_layers=6,
    conv_kernel_size=31,
    dropout=0.1,
    input_dropout=0.2,
    patch_size=14,
    patch_stride=4,
    gradient_checkpointing=False,  # Disable for testing
)

rnn_params = sum(p.numel() for p in rnn.parameters())
conf_params = sum(p.numel() for p in conformer.parameters())
print(f"RNN params: {rnn_params:,}")
print(f"Conformer params: {conf_params:,}")

rnn.eval()
conformer.eval()

# Same input
torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, neural_dim)
day_idx = torch.randint(0, n_days, (batch_size,))

print("\n" + "=" * 60)
print("Forward pass comparison")
print("=" * 60)

with torch.no_grad():
    rnn_out = rnn(x, day_idx)
    conf_out = conformer(x, day_idx)

print(f"Input shape: {x.shape}")
print(f"RNN output shape: {rnn_out.shape}")
print(f"Conformer output shape: {conf_out.shape}")

print(f"\nRNN logits stats:")
print(f"  mean: {rnn_out.mean():.4f}, std: {rnn_out.std():.4f}")
print(f"  min: {rnn_out.min():.4f}, max: {rnn_out.max():.4f}")

print(f"\nConformer logits stats:")
print(f"  mean: {conf_out.mean():.4f}, std: {conf_out.std():.4f}")
print(f"  min: {conf_out.min():.4f}, max: {conf_out.max():.4f}")

# Check softmax distribution (CTC expects reasonable probs)
rnn_probs = torch.softmax(rnn_out, dim=-1)
conf_probs = torch.softmax(conf_out, dim=-1)

print(f"\nRNN softmax entropy (avg): {-(rnn_probs * rnn_probs.log()).sum(-1).mean():.4f}")
print(f"Conformer softmax entropy (avg): {-(conf_probs * conf_probs.log()).sum(-1).mean():.4f}")
print(f"  (Max entropy for {n_classes} classes: {torch.log(torch.tensor(n_classes, dtype=torch.float)):.4f})")

# Check if outputs are degenerate
print(f"\nRNN - unique argmax values: {len(rnn_out.argmax(-1).unique())}")
print(f"Conformer - unique argmax values: {len(conf_out.argmax(-1).unique())}")

print("\n" + "=" * 60)
print("Layer-by-layer activation stats (Conformer)")
print("=" * 60)

# Trace through conformer manually
with torch.no_grad():
    # Day layer
    day_weights = conformer.day_weights[day_idx]
    day_biases = conformer.day_biases[day_idx]
    x_day = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
    x_day = conformer.day_layer_activation(x_day)
    print(f"After day layer: mean={x_day.mean():.4f}, std={x_day.std():.4f}")

    # Patch embedding
    x_patch = x_day.unsqueeze(1).permute(0, 3, 1, 2)
    x_unfold = x_patch.unfold(3, conformer.patch_size, conformer.patch_stride)
    x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)
    x_flat = x_unfold.reshape(x_unfold.size(0), x_unfold.size(1), -1)
    print(f"After patch flatten: mean={x_flat.mean():.4f}, std={x_flat.std():.4f}")

    # Input projection
    x_proj = conformer.input_projection(x_flat)
    print(f"After input projection: mean={x_proj.mean():.4f}, std={x_proj.std():.4f}")

    # Positional encoding
    x_pos = conformer.pos_encoding(x_proj)
    print(f"After positional encoding: mean={x_pos.mean():.4f}, std={x_pos.std():.4f}")

    # Through conformer blocks
    x_block = x_pos
    for i, block in enumerate(conformer.conformer_blocks):
        x_block = block(x_block)
        print(f"After conformer block {i}: mean={x_block.mean():.4f}, std={x_block.std():.4f}")

    # Output
    logits = conformer.output(x_block)
    print(f"Final logits: mean={logits.mean():.4f}, std={logits.std():.4f}")

print("\n" + "=" * 60)
print("Layer-by-layer activation stats (RNN)")
print("=" * 60)

with torch.no_grad():
    # Day layer
    day_weights_rnn = torch.stack([rnn.day_weights[i] for i in day_idx], dim=0)
    day_biases_rnn = torch.cat([rnn.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
    x_day_rnn = torch.einsum("btd,bdk->btk", x, day_weights_rnn) + day_biases_rnn
    x_day_rnn = rnn.day_layer_activation(x_day_rnn)
    print(f"After day layer: mean={x_day_rnn.mean():.4f}, std={x_day_rnn.std():.4f}")

    # Patch embedding
    x_patch_rnn = x_day_rnn.unsqueeze(1).permute(0, 3, 1, 2)
    x_unfold_rnn = x_patch_rnn.unfold(3, rnn.patch_size, rnn.patch_stride)
    x_unfold_rnn = x_unfold_rnn.squeeze(2).permute(0, 2, 3, 1)
    x_flat_rnn = x_unfold_rnn.reshape(x_unfold_rnn.size(0), x_unfold_rnn.size(1), -1)
    print(f"After patch flatten: mean={x_flat_rnn.mean():.4f}, std={x_flat_rnn.std():.4f}")

    # GRU
    states = rnn.h0.expand(rnn.n_layers, x.shape[0], rnn.n_units).contiguous()
    gru_out, _ = rnn.gru(x_flat_rnn, states)
    print(f"After GRU: mean={gru_out.mean():.4f}, std={gru_out.std():.4f}")

    # Output
    logits_rnn = rnn.out(gru_out)
    print(f"Final logits: mean={logits_rnn.mean():.4f}, std={logits_rnn.std():.4f}")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print("Compare the activation statistics above.")
print("Look for:")
print("  - Exploding/vanishing activations (very large or near-zero std)")
print("  - Mean drift (mean far from 0)")
print("  - Entropy differences (low entropy = model is too confident)")
