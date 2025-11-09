"""Multi-head attention mechanism for Transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for Transformer models."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Parameters
        ----------
        d_model : int, default=256
            Model dimension
        n_heads : int, default=8
            Number of attention heads
        dropout : float, default=0.1
            Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)
        attention_mask : Optional[torch.Tensor]
            Attention mask of shape (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

        # Compute Q, K, V
        Q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(scores.device)
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention_mask to proper shape
            mask = attention_mask.unsqueeze(1).unsqueeze(
                2
            )  # (batch_size, 1, 1, seq_len)
            mask = mask.expand(batch_size, self.n_heads, seq_len, seq_len)
            scores.masked_fill_(mask == 0, float("-inf"))

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, V)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final projection
        output = self.out_proj(attn_output)

        return output
