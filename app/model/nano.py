"""Complete GPT model implementation with transformer components."""

import torch
import torch.nn as nn
from typing import Optional
from .transformer import TransformerBlock
from .positional import PositionalEncoding


class NanoLLM(nn.Module):
    """Complete NanoLLM model with embedding, transformer blocks, and output projection."""

    def __init__(
        self,
        vocab_size: int = 10000,
        max_seq_len: int = 128,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        weight_tying: bool = True,
        pad_token_id: int = 0,
    ):
        """
        Initialize GPT model.

        Parameters
        ----------
        vocab_size : int, default=10000
            Size of vocabulary
        max_seq_len : int, default=128
            Maximum sequence length
        d_model : int, default=256
            Model dimension
        n_layers : int, default=4
            Number of transformer layers
        n_heads : int, default=8
            Number of attention heads
        d_ff : int, default=1024
            Feed-forward dimension
        dropout : float, default=0.1
            Dropout probability
        weight_tying : bool, default=True
            Whether to tie embedding and output weights
        pad_token_id : int, default=0
            ID of padding token
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.weight_tying = weight_tying
        self.pad_token_id = pad_token_id

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.token_embedding.padding_idx = pad_token_id

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        if weight_tying:
            self.output_projection.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_token_id].fill_(0)

        # Initialize output projection
        if not self.weight_tying:
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of GPT model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape (batch_size, seq_len)
        attention_mask : Optional[torch.Tensor]
            Attention mask of shape (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Positional encoding
        x = self.pos_encoding(x)

        # Apply dropout
        x = nn.Dropout(self.dropout)(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Output projection
        logits = self.output_projection(x)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape (batch_size, seq_len)
        max_new_tokens : int, default=50
            Maximum number of new tokens to generate
        temperature : float, default=1.0
            Sampling temperature
        do_sample : bool, default=True
            Whether to use sampling or greedy decoding
        top_k : Optional[int], default=None
            Top-k filtering parameter
        top_p : float, default=1.0
            Top-p (nucleus) filtering parameter
        repetition_penalty : float, default=1.0
            Penalty for repeating tokens (>1.0 penalizes, <1.0 encourages repetition)

        Returns
        -------
        torch.Tensor
            Generated token IDs of shape (batch_size, seq_len + new_tokens)
        """
        self.eval()
        with torch.no_grad():
            batch_size, seq_len = input_ids.size()
            current_ids = input_ids.clone()

            for _ in range(max_new_tokens):
                # Get model predictions
                logits = self.forward(current_ids)

                # Focus on last token
                logits = logits[:, -1, :] / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    # Create a score for each token based on how many times it appears in the generated sequence
                    for batch_idx in range(batch_size):
                        # Count token occurrences in the current sequence
                        unique_tokens, counts = torch.unique(
                            current_ids[batch_idx], return_counts=True
                        )
                        for token_id, count in zip(unique_tokens, counts):
                            if token_id < logits.size(-1):  # Ensure token is in vocab
                                # Apply penalty: logit / count^penalty
                                logits[batch_idx, token_id] = logits[
                                    batch_idx, token_id
                                ] / (count**repetition_penalty)

                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, top_k)
                    min_values = values[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < min_values,
                        torch.full_like(logits, float("-inf")),
                        logits,
                    )

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)

                    # Compute cumulative probabilities
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = torch.where(
                        indices_to_remove,
                        torch.full_like(logits, float("-inf")),
                        logits,
                    )

                # Apply softmax
                probs = torch.softmax(logits, dim=-1)

                # Sample or take argmax
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)

                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)

                # Stop if all sequences have generated EOS token
                if (next_token == 3).all():  # Assuming token 3 is EOS
                    break

            return current_ids
