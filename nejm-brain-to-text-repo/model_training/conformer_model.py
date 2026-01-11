"""
Conformer Decoder for Brain-to-Text

Replaces the GRU encoder with a Causal Conformer for improved phoneme error rate.
Maintains day-specific input layers for handling neural signal drift across sessions.

Architecture:
    Neural Input (512) → Day-Specific Layers → Patch Embedding → Causal Conformer → CTC Head
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer-based models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalConvolutionModule(nn.Module):
    """Causal convolution module for Conformer.

    Uses left-padding to ensure causality - each output only depends on
    current and previous inputs.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.kernel_size = kernel_size

        # Pointwise conv to expand channels, then GLU
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, bias=True
        )

        # Causal depthwise conv - pad only on left side
        self.depthwise_conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=0,  # We'll do manual causal padding
            groups=channels,
            bias=True
        )

        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, time, channels)
        Returns:
            Output tensor (batch, time, channels)
        """
        x = x.transpose(1, 2)  # (batch, channels, time)

        # Pointwise conv + GLU
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # Causal padding: pad only on the left
        x = nn.functional.pad(x, (self.kernel_size - 1, 0))
        x = self.depthwise_conv(x)

        # Norm + activation
        x = x.transpose(1, 2)  # (batch, time, channels)
        x = self.norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # (batch, channels, time)

        # Pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x.transpose(1, 2))  # (batch, time, channels)

        return x


class CausalMultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    Uses a causal mask to ensure each position can only attend to
    previous positions (including itself).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional padding mask (batch, seq_len)
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Create causal mask (upper triangular = -inf)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply padding mask if provided
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            padding_mask = ~mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.SiLU()  # Swish
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class ConformerBlock(nn.Module):
    """Single Conformer block with macaron-style feed-forward.

    Structure:
        x → FFN(0.5) → Self-Attention → Conv → FFN(0.5) → LayerNorm → out

    All sub-layers have residual connections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        # First feed-forward (half-step)
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1 = FeedForward(d_model, d_ff, dropout)

        # Self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Convolution
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = CausalConvolutionModule(d_model, conv_kernel_size, dropout)

        # Second feed-forward (half-step)
        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2 = FeedForward(d_model, d_ff, dropout)

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional padding mask (batch, seq_len)
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # First FFN (half-step)
        x = x + 0.5 * self.ff1(self.ff1_norm(x))

        # Self-attention
        x = x + self.attn_dropout(self.attn(self.attn_norm(x), mask))

        # Convolution
        x = x + self.conv(self.conv_norm(x))

        # Second FFN (half-step)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))

        # Final norm
        x = self.final_norm(x)

        return x


class ConformerDecoder(nn.Module):
    """
    Causal Conformer decoder for brain-to-text.

    Replaces the GRU with a stack of Conformer blocks while maintaining
    the day-specific input layers for handling neural signal drift.

    Architecture:
        Neural (512) → Day Layer → Patch Embed → Conformer Blocks → CTC Head
    """

    def __init__(
        self,
        neural_dim: int,
        n_days: int,
        n_classes: int,
        d_model: int = 768,
        n_heads: int = 8,
        d_ff: int = 3072,
        n_layers: int = 6,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        input_dropout: float = 0.2,
        patch_size: int = 14,
        patch_stride: int = 4,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            neural_dim: Number of input neural features (512)
            n_days: Number of recording sessions
            n_classes: Number of output classes (41 phonemes)
            d_model: Conformer hidden dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            n_layers: Number of Conformer blocks
            conv_kernel_size: Kernel size for convolution module
            dropout: Dropout rate for Conformer
            input_dropout: Dropout rate for input layer
            patch_size: Number of timesteps to concatenate
            patch_stride: Stride for patch embedding
            gradient_checkpointing: Use gradient checkpointing to save memory
        """
        super().__init__()

        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.gradient_checkpointing = gradient_checkpointing

        # Day-specific input layers (same as GRUDecoder)
        self.day_layer_activation = nn.Softsign()
        # Use stacked tensors instead of ParameterList for torch.compile compatibility
        self.day_weights = nn.Parameter(
            torch.stack([torch.eye(neural_dim) for _ in range(n_days)], dim=0)
        )  # Shape: (n_days, neural_dim, neural_dim)
        self.day_biases = nn.Parameter(
            torch.zeros(n_days, 1, neural_dim)
        )  # Shape: (n_days, 1, neural_dim)
        self.day_layer_dropout = nn.Dropout(input_dropout)
        self.input_dropout = input_dropout

        # Input projection: patch_size * neural_dim → d_model
        input_size = neural_dim * patch_size
        self.input_projection = nn.Linear(input_size, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000, dropout=dropout)

        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # CTC output head
        self.output = nn.Linear(d_model, n_classes)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Neural features (batch, time, neural_dim)
            day_idx: Day indices for each sample in batch (batch,)
            mask: Optional padding mask (batch, time)
        Returns:
            Logits (batch, time', n_classes) where time' is reduced by patching
        """
        # Apply day-specific transformation (tensor indexing for torch.compile compatibility)
        day_weights = self.day_weights[day_idx]  # (batch, neural_dim, neural_dim)
        day_biases = self.day_biases[day_idx]    # (batch, 1, neural_dim)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # Patch embedding (same as GRUDecoder)
        if self.patch_size > 0:
            x = x.unsqueeze(1)  # (batch, 1, time, features)
            x = x.permute(0, 3, 1, 2)  # (batch, features, 1, time)

            # Extract patches using unfold
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2)  # (batch, features, n_patches, patch_size)
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # (batch, n_patches, patch_size, features)

            # Flatten patch
            x = x_unfold.reshape(x_unfold.size(0), x_unfold.size(1), -1)

        # Project to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply Conformer blocks
        for block in self.conformer_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)

        # Output projection
        logits = self.output(x)

        return logits


def count_parameters(model: nn.Module) -> dict:
    """Count trainable parameters by component."""
    counts = {
        'day_layers': 0,
        'input_projection': 0,
        'conformer': 0,
        'output': 0,
        'total': 0,
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n_params = param.numel()
        counts['total'] += n_params

        if 'day' in name:
            counts['day_layers'] += n_params
        elif 'input_projection' in name or 'pos_encoding' in name:
            counts['input_projection'] += n_params
        elif 'output' in name:
            counts['output'] += n_params
        else:
            counts['conformer'] += n_params

    return counts


if __name__ == "__main__":
    # Test the model
    model = ConformerDecoder(
        neural_dim=512,
        n_days=46,
        n_classes=41,
        d_model=768,
        n_heads=8,
        d_ff=3072,
        n_layers=6,
        conv_kernel_size=31,
        dropout=0.1,
        input_dropout=0.2,
        patch_size=14,
        patch_stride=4,
    )

    # Print parameter counts
    counts = count_parameters(model)
    print("Parameter counts:")
    for key, value in counts.items():
        print(f"  {key}: {value:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 200
    x = torch.randn(batch_size, seq_len, 512)
    day_idx = torch.randint(0, 46, (batch_size,))

    logits = model(x, day_idx)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Expected output time: (200 - 14) / 4 + 1 = 47
    expected_time = (seq_len - 14) // 4 + 1
    print(f"Expected output time: {expected_time}")
