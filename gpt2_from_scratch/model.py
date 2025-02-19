import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Model Architecture Hyperparameters
context_length = 128  # Maximum sequence length the model can process
d_model = 512        # Embedding dimension size (hidden state size)
num_block = 12       # Number of transformer blocks in the model
num_heads = 8        # Number of attention heads for multi-head attention
dropout = 0.1        # Dropout rate for regularization

# Set device priority: CUDA GPU > Apple M1/M2 > CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Set random seed for reproducibility
TOUCH_SEED = 1337
torch.manual_seed(TOUCH_SEED)

class FeedForwardNetwork(nn.Module):
    """Feed-forward network following attention layer in transformer block.
    Consists of two linear transformations with a ReLU activation in between."""
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),     # First linear layer expands dimension
            nn.ReLU(),                           # ReLU activation
            nn.Linear(4 * d_model, d_model),     # Second linear layer projects back to model dimension
            nn.Dropout(dropout),                 # Dropout for regularization
        )

    def forward(self, x):
        return self.ffn(x)

class Attention(nn.Module):
    """Single head of self-attention mechanism."""
    def __init__(self):
        super().__init__()
        # Linear projections for query, key, and value
        self.Wq = nn.Linear(d_model, d_model // num_heads, bias=False)
        self.Wk = nn.Linear(d_model, d_model // num_heads, bias=False)
        self.Wv = nn.Linear(d_model, d_model // num_heads, bias=False)
        # Causal mask to prevent attending to future tokens
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape  # batch_size, sequence_length, channels
        # Project input into query, key, value vectors
        q = self.Wq(x)    # queries
        k = self.Wk(x)    # keys
        v = self.Wv(x)    # values
        
        # Compute attention scores
        weights = (q @ k.transpose(-2, -1)) *  math.sqrt(d_model // num_heads)
        # Apply causal mask to prevent attending to future tokens
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        # Compute weighted sum of values
        output = weights @ v
        return output

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism that runs multiple attention heads in parallel."""
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.projection_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Run attention heads in parallel and concatenate results
        head_outputs = [head(x) for head in self.heads]
        head_outputs = torch.cat(head_outputs, dim=-1)
        # Project concatenated outputs back to model dimension
        out = self.dropout(self.projection_layer(head_outputs))
        return out

class TransformerBlock(nn.Module):
    """Single transformer block combining multi-head attention and feed-forward network."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # Layer norm before attention
        self.ln2 = nn.LayerNorm(d_model)  # Layer norm before FFN
        
        self.mha = MultiHeadAttention()   # Multi-head attention
        self.ffn = FeedForwardNetwork()   # Feed-forward network
        
    def forward(self, x):
        # Apply attention with residual connection
        x = x + self.mha(self.ln1(x))
        # Apply FFN with residual connection
        x = x + self.ffn(self.ln2(x))
        return x

class Model(nn.Module):
    """Complete transformer model for language modeling."""
    def __init__(self, max_token_value=100080):
        super().__init__()
        # Token embedding layer
        self.token_embedding_lookup_table = nn.Embedding(max_token_value, d_model)
        # Stack of transformer blocks with final layer norm
        self.transformer_blocks = nn.Sequential(*(
          [TransformerBlock() for _ in range(num_block)] +
          [nn.LayerNorm(d_model)]
        ))
        # Output projection to vocabulary size
        self.model_out_linear_layer = nn.Linear(d_model, max_token_value)
    
    def forward(self, idx, targets=None):
        """Forward pass of the model."""
        B, T = idx.shape
        
        # Create positional encodings using sine and cosine functions
        position_encoding_lookup_table = torch.zeros(context_length, d_model, device=device)
        position = torch.arange(0, context_length, dtype=torch.long).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        
        # Combine token embeddings and positional encodings
        x = self.token_embedding_lookup_table(idx) + position_embedding
        # Pass through transformer blocks
        x = self.transformer_blocks(x)
        # Project to vocabulary size
        logits = self.model_out_linear_layer(x)
        
        # Calculate loss if targets are provided
        if targets is not None:
            B, T, C = logits.shape
            logits.reshaped = logits.view(B * T, C)
            targets.reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits.reshaped, target=targets.reshaped)
        else:
            loss = None
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens=100):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop sequence to maximum context length
            idx_crop = idx[:, -context_length:]
            # Get predictions
            logits, loss = self.forward(idx_crop)
            # Focus on last timestep
            logits_last_timestep = logits[:, -1, :]
            # Get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample next token
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx
