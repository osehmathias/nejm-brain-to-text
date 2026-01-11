"""
Debug script to trace variance propagation through all layers at initialization.
Covers checklist items: 1.4, 1.5, 3.1, 3.2, 3.3, 3.4, 3.5

This helps identify if any layer causes variance explosion or collapse.
"""
import torch
import torch.nn as nn
import sys
import math
sys.path.insert(0, '/Users/oseh/code/river/research/nejm-brain-to-text/nejm-brain-to-text-repo/model_training')

from rnn_model import GRUDecoder
from conformer_model import ConformerDecoder

# Params matching training config
neural_dim = 512
n_days = 45
n_classes = 41
batch_size = 32
seq_len = 200

print("=" * 70)
print("INIT VARIANCE ANALYSIS")
print("=" * 70)
print("Tracing variance (std^2) through each layer at initialization")
print("Target: std should stay roughly in [0.5, 2.0] range throughout\n")

# Create models
torch.manual_seed(42)
rnn = GRUDecoder(
    neural_dim=neural_dim, n_units=768, n_days=n_days, n_classes=n_classes,
    rnn_dropout=0.0, input_dropout=0.0, n_layers=5, patch_size=14, patch_stride=4,
)

torch.manual_seed(42)
conformer = ConformerDecoder(
    neural_dim=neural_dim, n_days=n_days, n_classes=n_classes,
    d_model=768, n_heads=8, d_ff=3072, n_layers=6, conv_kernel_size=31,
    dropout=0.0, input_dropout=0.0, patch_size=14, patch_stride=4,
    gradient_checkpointing=False,
)

rnn.eval()
conformer.eval()

# Input
torch.manual_seed(123)
x = torch.randn(batch_size, seq_len, neural_dim)
day_idx = torch.randint(0, n_days, (batch_size,))

print("=" * 70)
print("CONFORMER LAYER-BY-LAYER VARIANCE")
print("=" * 70)

with torch.no_grad():
    print(f"{'Layer':<45} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 85)

    print(f"{'Input':<45} {x.mean().item():>10.4f} {x.std().item():>10.4f} {x.min().item():>10.4f} {x.max().item():>10.4f}")

    # Day layer
    day_weights = conformer.day_weights[day_idx]
    day_biases = conformer.day_biases[day_idx]
    x_day = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
    print(f"{'Day layer (pre-activation)':<45} {x_day.mean().item():>10.4f} {x_day.std().item():>10.4f} {x_day.min().item():>10.4f} {x_day.max().item():>10.4f}")

    x_day = conformer.day_layer_activation(x_day)
    print(f"{'Day layer (post-softsign)':<45} {x_day.mean().item():>10.4f} {x_day.std().item():>10.4f} {x_day.min().item():>10.4f} {x_day.max().item():>10.4f}")

    # Patch embedding
    x_patch = x_day.unsqueeze(1).permute(0, 3, 1, 2)
    x_unfold = x_patch.unfold(3, conformer.patch_size, conformer.patch_stride)
    x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)
    x_flat = x_unfold.reshape(x_unfold.size(0), x_unfold.size(1), -1)
    print(f"{'Patch flatten (14*512=7168 dim)':<45} {x_flat.mean().item():>10.4f} {x_flat.std().item():>10.4f} {x_flat.min().item():>10.4f} {x_flat.max().item():>10.4f}")

    # Input projection
    x_proj = conformer.input_projection(x_flat)
    print(f"{'Input projection (768 dim)':<45} {x_proj.mean().item():>10.4f} {x_proj.std().item():>10.4f} {x_proj.min().item():>10.4f} {x_proj.max().item():>10.4f}")

    # Check input projection weight stats
    w = conformer.input_projection.weight
    print(f"  -> weight stats: mean={w.mean().item():.6f}, std={w.std().item():.6f}")
    expected_xavier_std = math.sqrt(2.0 / (7168 + 768))
    print(f"  -> expected Xavier std: {expected_xavier_std:.6f}")

    # Positional encoding
    x_pos = conformer.pos_encoding(x_proj)
    print(f"{'Positional encoding added':<45} {x_pos.mean().item():>10.4f} {x_pos.std().item():>10.4f} {x_pos.min().item():>10.4f} {x_pos.max().item():>10.4f}")

    # Check PE magnitude
    pe_sample = conformer.pos_encoding.pe[:, :x_proj.size(1), :]
    print(f"  -> PE alone: mean={pe_sample.mean().item():.4f}, std={pe_sample.std().item():.4f}")

    # Through conformer blocks
    x_block = x_pos
    for i, block in enumerate(conformer.conformer_blocks):
        # Detailed trace through block
        x_in = x_block

        # FFN1
        x_ffn1 = block.ff1_norm(x_in)
        x_ffn1_out = block.ff1(x_ffn1)
        x_after_ffn1 = x_in + 0.5 * x_ffn1_out

        # Attention
        x_attn_norm = block.attn_norm(x_after_ffn1)
        x_attn_out = block.attn(x_attn_norm)
        x_after_attn = x_after_ffn1 + x_attn_out

        # Conv
        x_conv_norm = block.conv_norm(x_after_attn)
        x_conv_out = block.conv(x_conv_norm)
        x_after_conv = x_after_attn + x_conv_out

        # FFN2
        x_ffn2_norm = block.ff2_norm(x_after_conv)
        x_ffn2_out = block.ff2(x_ffn2_norm)
        x_after_ffn2 = x_after_conv + 0.5 * x_ffn2_out

        # Final norm
        x_block = block.final_norm(x_after_ffn2)

        print(f"{'Conformer Block ' + str(i) + ' output':<45} {x_block.mean().item():>10.4f} {x_block.std().item():>10.4f} {x_block.min().item():>10.4f} {x_block.max().item():>10.4f}")

        # Print sub-component contributions for first block
        if i == 0:
            print(f"  -> FFN1 residual contribution std: {(0.5 * x_ffn1_out).std().item():.4f}")
            print(f"  -> Attn residual contribution std: {x_attn_out.std().item():.4f}")
            print(f"  -> Conv residual contribution std: {x_conv_out.std().item():.4f}")
            print(f"  -> FFN2 residual contribution std: {(0.5 * x_ffn2_out).std().item():.4f}")

    # Output layer
    logits = conformer.output(x_block)
    print(f"{'Output logits':<45} {logits.mean().item():>10.4f} {logits.std().item():>10.4f} {logits.min().item():>10.4f} {logits.max().item():>10.4f}")

    # Output weight stats
    w_out = conformer.output.weight
    print(f"  -> output weight stats: mean={w_out.mean().item():.6f}, std={w_out.std().item():.6f}")
    print(f"  -> expected std for logit std=0.07: {0.07 / math.sqrt(768):.6f}")

print("\n" + "=" * 70)
print("RNN LAYER-BY-LAYER VARIANCE (for comparison)")
print("=" * 70)

with torch.no_grad():
    print(f"{'Layer':<45} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 85)

    # Reset input
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, neural_dim)

    print(f"{'Input':<45} {x.mean().item():>10.4f} {x.std().item():>10.4f} {x.min().item():>10.4f} {x.max().item():>10.4f}")

    # Day layer
    day_weights_rnn = torch.stack([rnn.day_weights[i] for i in day_idx], dim=0)
    day_biases_rnn = torch.cat([rnn.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
    x_day_rnn = torch.einsum("btd,bdk->btk", x, day_weights_rnn) + day_biases_rnn
    print(f"{'Day layer (pre-activation)':<45} {x_day_rnn.mean().item():>10.4f} {x_day_rnn.std().item():>10.4f} {x_day_rnn.min().item():>10.4f} {x_day_rnn.max().item():>10.4f}")

    x_day_rnn = rnn.day_layer_activation(x_day_rnn)
    print(f"{'Day layer (post-softsign)':<45} {x_day_rnn.mean().item():>10.4f} {x_day_rnn.std().item():>10.4f} {x_day_rnn.min().item():>10.4f} {x_day_rnn.max().item():>10.4f}")

    # Patch embedding
    x_patch_rnn = x_day_rnn.unsqueeze(1).permute(0, 3, 1, 2)
    x_unfold_rnn = x_patch_rnn.unfold(3, rnn.patch_size, rnn.patch_stride)
    x_unfold_rnn = x_unfold_rnn.squeeze(2).permute(0, 2, 3, 1)
    x_flat_rnn = x_unfold_rnn.reshape(x_unfold_rnn.size(0), x_unfold_rnn.size(1), -1)
    print(f"{'Patch flatten (14*512=7168 dim)':<45} {x_flat_rnn.mean().item():>10.4f} {x_flat_rnn.std().item():>10.4f} {x_flat_rnn.min().item():>10.4f} {x_flat_rnn.max().item():>10.4f}")

    # GRU
    states = rnn.h0.expand(rnn.n_layers, batch_size, rnn.n_units).contiguous()
    print(f"{'GRU initial hidden state (h0)':<45} {states.mean().item():>10.4f} {states.std().item():>10.4f} {states.min().item():>10.4f} {states.max().item():>10.4f}")

    gru_out, _ = rnn.gru(x_flat_rnn, states)
    print(f"{'GRU output':<45} {gru_out.mean().item():>10.4f} {gru_out.std().item():>10.4f} {gru_out.min().item():>10.4f} {gru_out.max().item():>10.4f}")

    # Output
    logits_rnn = rnn.out(gru_out)
    print(f"{'Output logits':<45} {logits_rnn.mean().item():>10.4f} {logits_rnn.std().item():>10.4f} {logits_rnn.min().item():>10.4f} {logits_rnn.max().item():>10.4f}")

print("\n" + "=" * 70)
print("ATTENTION ANALYSIS (Item 3.2)")
print("=" * 70)

with torch.no_grad():
    # Get attention weights from first block
    block = conformer.conformer_blocks[0]

    # Prepare input
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, neural_dim)

    # Forward to get to attention input
    day_weights = conformer.day_weights[day_idx]
    day_biases = conformer.day_biases[day_idx]
    x_day = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
    x_day = conformer.day_layer_activation(x_day)

    x_patch = x_day.unsqueeze(1).permute(0, 3, 1, 2)
    x_unfold = x_patch.unfold(3, conformer.patch_size, conformer.patch_stride)
    x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)
    x_flat = x_unfold.reshape(x_unfold.size(0), x_unfold.size(1), -1)

    x_proj = conformer.input_projection(x_flat)
    x_pos = conformer.pos_encoding(x_proj)

    # Through FFN1
    x_ffn1 = x_pos + 0.5 * block.ff1(block.ff1_norm(x_pos))

    # Get attention scores
    attn = block.attn
    x_norm = block.attn_norm(x_ffn1)

    batch_size_local, seq_len_local, _ = x_norm.shape
    q = attn.w_q(x_norm).view(batch_size_local, seq_len_local, attn.n_heads, attn.d_k).transpose(1, 2)
    k = attn.w_k(x_norm).view(batch_size_local, seq_len_local, attn.n_heads, attn.d_k).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.d_k)

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len_local, seq_len_local, dtype=torch.bool), diagonal=1)
    scores_masked = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weights = torch.softmax(scores_masked, dim=-1)

    print(f"Attention head dim (d_k): {attn.d_k}")
    print(f"Pre-softmax scores - mean: {scores[~causal_mask.unsqueeze(0).unsqueeze(0).expand_as(scores)].mean():.4f}, std: {scores[~causal_mask.unsqueeze(0).unsqueeze(0).expand_as(scores)].std():.4f}")
    print(f"Post-softmax weights - max: {attn_weights.max():.4f}, entropy: {-(attn_weights * (attn_weights + 1e-10).log()).sum(-1).mean():.4f}")
    print(f"Expected uniform entropy for seq_len={seq_len_local}: {math.log(seq_len_local / 2):.4f} (avg)")

print("\n" + "=" * 70)
print("CONVOLUTION MODULE ANALYSIS (Item 3.3)")
print("=" * 70)

with torch.no_grad():
    conv = conformer.conformer_blocks[0].conv

    print(f"Pointwise conv1 weight std: {conv.pointwise_conv1.weight.std():.6f}")
    print(f"Depthwise conv weight std: {conv.depthwise_conv.weight.std():.6f}")
    print(f"Pointwise conv2 weight std: {conv.pointwise_conv2.weight.std():.6f}")

    # Expected init
    print(f"\nExpected Kaiming init for conv1 (fan_in={conv.pointwise_conv1.in_channels}): {math.sqrt(2.0 / conv.pointwise_conv1.in_channels):.6f}")

print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

# Compute key metrics
with torch.no_grad():
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, neural_dim)

    conf_out = conformer(x, day_idx)
    rnn_out = rnn(x, day_idx)

    conf_entropy = -(torch.softmax(conf_out, -1) * torch.log_softmax(conf_out, -1)).sum(-1).mean()
    rnn_entropy = -(torch.softmax(rnn_out, -1) * torch.log_softmax(rnn_out, -1)).sum(-1).mean()
    max_entropy = math.log(n_classes)

    print(f"Conformer output std: {conf_out.std():.4f} (target: ~0.07)")
    print(f"RNN output std: {rnn_out.std():.4f}")
    print(f"\nConformer entropy: {conf_entropy:.4f} / {max_entropy:.4f} ({conf_entropy/max_entropy*100:.1f}% of max)")
    print(f"RNN entropy: {rnn_entropy:.4f} / {max_entropy:.4f} ({rnn_entropy/max_entropy*100:.1f}% of max)")

    if conf_out.std() > 0.15:
        print("\n[WARNING] Conformer output std still too high - output init may not be applied")
    if conf_entropy < 0.9 * max_entropy:
        print("[WARNING] Conformer entropy below 90% of max - softmax too peaked at init")
