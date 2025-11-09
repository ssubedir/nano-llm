"""Position-wise feed-forward network for Transformer blocks."""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network for Transformer blocks."""

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Parameters
        ----------
        d_model : int, default=256
            Model dimension
        d_ff : int, default=1024
            Feed-forward dimension
        dropout : float, default=0.1
            Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.net(x)
