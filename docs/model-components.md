# Model Components Documentation

This document provides detailed technical information about the model components in nano-llm.
## Overview

nano-llm implements a simple GPT-style transformer architecture with the following core components:

1. **NanoLLM**: The main model class that combines all components
2. **TransformerBlock**: The fundamental transformer layer
3. **MultiHeadAttention**: The attention mechanism
4. **PositionwiseFeedForward**: The position-wise feed-forward network
5. **PositionalEncoding**: Positional information for tokens

## NanoLLM Model

### Architecture

The NanoLLM model follows the decoder-only transformer architecture:

```
Input Tokens → Token Embedding → Positional Encoding → N × Transformer Blocks → Output Projection → Logits
```

### Implementation Details

```python
class NanoLLM(nn.Module):
    def __init__(self,
                 vocab_size: int = 10000,
                 max_seq_len: int = 128,
                 d_model: int = 256,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 weight_tying: bool = True,
                 pad_token_id: int = 0):
```

#### Key Parameters

- **vocab_size**: Size of the vocabulary (number of unique tokens)
- **max_seq_len**: Maximum sequence length the model can handle
- **d_model**: Dimensionality of the model's hidden states
- **n_layers**: Number of transformer blocks in the stack
- **n_heads**: Number of attention heads in each transformer block
- **d_ff**: Dimensionality of the feed-forward network
- **dropout**: Dropout probability for regularization
- **weight_tying**: Whether to tie input and output embedding weights
- **pad_token_id**: Token ID used for padding sequences

### Forward Pass

The forward pass processes input sequences through the model:

```python
def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
    # 1. Token embeddings
    # 2. Positional encoding
    # 3. Apply dropout
    # 4. Pass through transformer layers
    # 5. Output projection
    return logits
```

### Weight Tying

When `weight_tying=True`, the input embedding and output projection matrices share weights:

**Benefits:**
- Reduces model parameters
- Improves generalization
- Prevents embedding matrix from becoming inconsistent

**Mathematical Formulation:**
```
Output_Weight = Input_Weight^T
```

## Multi-Head Attention

### Mathematical Foundation

Multi-head attention allows the model to focus on different positions of the input sequence:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q (Query): What I'm looking for
- K (Key): What I can offer
- V (Value): What I actually offer

### Implementation

The attention mechanism is implemented with the following steps:

1. **Linear Projections**: Project input to Q, K, V matrices
2. **Scaled Dot-Product**: Compute attention scores
3. **Softmax**: Normalize scores to probabilities
4. **Value Weighting**: Weight values by attention probabilities
5. **Concatenation**: Combine multiple heads
6. **Final Projection**: Linear layer to combine heads

### Key Features

- **Causal Masking**: Ensures tokens only attend to previous positions
- **Multi-Head Design**: Allows parallel attention to different representation subspaces
- **Efficient Implementation**: Optimized for GPU computation

### Attention Head Computation

For each attention head h:

```
Q_h = XW_h^Q
K_h = XW_h^K
V_h = XW_h^V
Attention_h = Attention(Q_h, K_h, V_h)
```

Where X is the input and W are learnable weight matrices.

## Positionwise Feed-Forward Network

### Architecture

Each transformer block contains a position-wise feed-forward network:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

### Implementation Details

1. **First Linear Layer**: Expands dimension from d_model to d_ff
2. **ReLU Activation**: Introduces non-linearity
3. **Second Linear Layer**: Projects back to d_model
4. **Dropout**: Applied after each linear layer

### Dimensionality

The feed-forward network typically uses an expansion factor of 4:
- Input dimension: d_model
- Hidden dimension: d_ff = 4 × d_model
- Output dimension: d_model

## Positional Encoding (TODO: RoPE)

### Purpose

Since transformers don't have inherent notion of sequence order, positional encodings add position information:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Implementation

The sinusoidal positional encoding has several advantages:
- **Deterministic**: No learnable parameters
- **Generalizable**: Can handle sequences longer than seen during training
- **Unique**: Each position gets a unique encoding

### Alternative Approaches

While nano-llm currently uses sinusoidal encodings, other options include:
- **Learned Positional Embeddings**: Learned positional vectors
- **Relative Position Encodings**: Attend to relative positions
- **Rotary Position Encodings**: Rotary embeddings (RoPE)

## Transformer Block

### Architecture (TODO: Use RMSNorm)

Each transformer block combines attention and feed-forward layers:

```
x → LayerNorm → Multi-Head Attention → Add → LayerNorm → FFN → Add → Output
```

### Implementation Details

1. **Pre-Normalization**: Layer normalization before sub-layers
2. **Residual Connections**: Add input to output of each sub-layer
3. **Dropout**: Applied throughout for regularization

### Benefits of Pre-Norm

- **Training Stability**: More stable gradients
- **Better Performance**: Often converges faster
- **Simpler Tuning**: Less sensitive to hyperparameters

## Generation Process

### Autoregressive Generation

The model generates text autoregressively:

```
Input: "The cat"
Step 1: Predict next token → "ate"
Step 2: Predict next token → "the"
Step 3: Predict next token → "fish"
...
```

### Sampling Strategies

1. **Greedy Sampling**: Always pick the most likely token
2. **Top-k Sampling**: Limit to k most likely tokens
3. **Nucleus Sampling**: Use smallest set with cumulative probability ≥ p
4. **Temperature Scaling**: Control randomness

### Implementation

The generation process includes:
- **Sequence Management**: Handle sequence length limits
- **Causal Masking**: Ensure proper autoregressive behavior
- **Special Tokens**: Handle padding, EOS tokens appropriately

## Model Initialization

### Weight Initialization

Proper initialization is crucial for training stability:

```python
# Embeddings: Normal distribution with std=0.02
nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

# Padding token: Zero initialization
if self.pad_token_id is not None:
    with torch.no_grad():
        self.token_embedding.weight[self.pad_token_id].fill_(0)
```

### Why These Values?

- **Standard Deviation 0.02**: Empirically works well for transformers
- **Zero Padding**: Padding tokens shouldn't contribute to gradients
- **Weight Tying**: Reduces parameters and improves performance

## Next Steps

- Check the [Configuration Reference](configuration.md) for model hyperparameters