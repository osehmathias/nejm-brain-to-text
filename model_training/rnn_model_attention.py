import torch
from torch import nn

class GRUDecoderWithAttention(nn.Module):
    '''
    GRU decoder with self-attention mechanism

    This class combines day-specific input layers, a GRU, self-attention, and an output classification layer
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5,
                 patch_size = 0,
                 patch_stride = 0,
                 n_attention_heads = 8,
                 attention_dropout = 0.1,
                 attention_layers = 1,
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes
        rnn_dropout    (float) - percentage of units to dropout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input
        n_attention_heads (int) - number of attention heads
        attention_dropout (float) - dropout rate for attention
        attention_layers (int) - number of self-attention layers after GRU
        '''
        super(GRUDecoderWithAttention, self).__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.n_attention_heads = n_attention_heads
        self.attention_dropout = attention_dropout
        self.attention_layers = attention_layers

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout,
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Self-attention layers after GRU
        self.attention_blocks = nn.ModuleList()
        for _ in range(self.attention_layers):
            attention_block = nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=self.n_units,
                    num_heads=self.n_attention_heads,
                    dropout=self.attention_dropout,
                    batch_first=True,
                ),
                'norm1': nn.LayerNorm(self.n_units),
                'ffn': nn.Sequential(
                    nn.Linear(self.n_units, self.n_units * 4),
                    nn.GELU(),
                    nn.Dropout(self.attention_dropout),
                    nn.Linear(self.n_units * 4, self.n_units),
                    nn.Dropout(self.attention_dropout),
                ),
                'norm2': nn.LayerNorm(self.n_units),
            })
            self.attention_blocks.append(attention_block)

        # Prediction head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def _generate_causal_mask(self, seq_len, device):
        '''Generate a causal attention mask to maintain autoregressive property'''
        # Upper triangular matrix of -inf, diagonal and below are 0
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x.
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0:

            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]

            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]

            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Pass input through RNN
        output, hidden_states = self.gru(x, states)

        # Apply self-attention layers with causal masking
        seq_len = output.size(1)
        causal_mask = self._generate_causal_mask(seq_len, output.device)

        for block in self.attention_blocks:
            # Self-attention with residual connection and layer norm
            attn_out, _ = block['attention'](
                output, output, output,
                attn_mask=causal_mask,
                is_causal=True,
            )
            output = block['norm1'](output + attn_out)

            # Feed-forward with residual connection and layer norm
            ffn_out = block['ffn'](output)
            output = block['norm2'](output + ffn_out)

        # Compute logits
        logits = self.out(output)

        if return_state:
            return logits, hidden_states

        return logits
