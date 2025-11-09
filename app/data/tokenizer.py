"""Tokenizer implementation for training BPE tokenizers."""

from typing import List
import os


class TokenizerTrainer:
    """BPE tokenizer trainer."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size

    def train_tokenizer(self, texts: List[str], save_path: str):
        """Train a BPE tokenizer."""
        from tokenizers.implementations import ByteLevelBPETokenizer

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            texts,
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        )

        # Save tokenizer
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_model(save_path)

        return tokenizer
