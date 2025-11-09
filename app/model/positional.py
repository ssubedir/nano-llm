"""Positional encoding for Transformer models using sinusoidal functions."""

import torch
import torch.nn as nn
import math


# TODO: use Rotary positional encoding (RoPE) -- More Modern Alternatives to Positional Encoding for Transformer Models
class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models using sinusoidal functions."""

    def __init__(self, d_model: int = 256, max_seq_len: int = 128):
        """
        Initialize positional encoding.

        Parameters
        ----------
        d_model : int, default=256
            Model dimension
        max_seq_len : int, default=128
            Maximum sequence length
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # Calculate div_term for sin and cos
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # Apply sin to even positions, cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)

        Returns
        -------
        torch.Tensor
            Input with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
