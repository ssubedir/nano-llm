"""Complete Transformer block with attention and feed-forward layers."""

import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """Complete Transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Initialize Transformer block.

        Parameters
        ----------
        d_model : int, default=256
            Model dimension
        n_heads : int, default=8
            Number of attention heads
        d_ff : int, default=1024
            Feed-forward dimension
        dropout : float, default=0.1
            Dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # TODO: RMSNorm instead of LayerNorm for efficiency
        # Layer normalization (Pre-LN architecture for stability)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Transformer block.

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
        # Pre-LN Transformer architecture
        # First sub-layer: Multi-head attention with residual connection
        attn_output = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attn_output)

        # Second sub-layer: Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x
