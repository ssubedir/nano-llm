"""Dataset implementation using PyTorch Dataset and DataLoader."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional


class TextDataset(Dataset):
    """Dataset that loads and tokenizes text data."""

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_seq_len: int = 128,
        dynamic_padding: bool = False,
    ):
        """
        Initialize text dataset.

        Parameters
        ----------
        texts : List[str]
            List of input texts
        tokenizer : Tokenizer
            Trained tokenizer
        max_seq_len : int, default=128
            Maximum sequence length
        dynamic_padding : bool, default=False
            Whether to use dynamic padding (no padding in __getitem__)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dynamic_padding = dynamic_padding
        self.pad_token_id = self.tokenizer.token_to_id("<pad>") or 0

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode(text)

        # Truncate to max_seq_len
        input_ids = encoded.ids[: self.max_seq_len]
        original_length = len(input_ids)

        if self.dynamic_padding:
            # For dynamic padding, return unpadded sequence
            attention_mask = [1] * original_length
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "length": original_length,  # Store original length for collate function
            }
        else:
            # Static padding (original behavior)
            if len(input_ids) < self.max_seq_len:
                input_ids = input_ids + [self.pad_token_id] * (
                    self.max_seq_len - len(input_ids)
                )

            # Create attention mask
            attention_mask = [1] * original_length
            if len(attention_mask) < self.max_seq_len:
                attention_mask = attention_mask + [0] * (
                    self.max_seq_len - len(attention_mask)
                )

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }


def dynamic_padding_collate(batch, pad_token_id=0, pad_to_multiple=None):
    """
    Custom collate function for dynamic padding.

    Parameters
    ----------
    batch : List[Dict]
        List of samples from the dataset
    pad_token_id : int, default=0
        Token ID to use for padding
    pad_to_multiple : int, optional
        If provided, pad sequences to multiples of this value (for GPU efficiency)

    Returns
    -------
    Dict
        Batch with padded input_ids and attention_mask
    """
    # Get the maximum length in this batch
    max_len = max(item["length"] for item in batch)

    # Adjust for pad_to_multiple if specified
    if pad_to_multiple:
        max_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

    # Prepare batch tensors
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    # Fill the tensors
    for i, item in enumerate(batch):
        seq_len = item["length"]
        input_ids[i, :seq_len] = item["input_ids"][:seq_len]
        attention_mask[i, :seq_len] = item["attention_mask"][:seq_len]

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_collate_fn(pad_token_id, pad_to_multiple):
    """
    Create a collate function with fixed parameters.
    
    Parameters
    ----------
    pad_token_id : int
        Token ID to use for padding
    pad_to_multiple : int, optional
        If provided, pad sequences to multiples of this value
    
    Returns
    -------
    function
        Collate function that can be passed to DataLoader
    """
    def collate_fn(batch):
        return dynamic_padding_collate(batch, pad_token_id, pad_to_multiple)
    return collate_fn


def create_dataloaders(
    texts: List[str],
    tokenizer,
    batch_size: int = 16,
    max_seq_len: int = 128,
    train_split: float = 0.9,
    dynamic_padding: bool = False,
    pad_to_multiple: Optional[int] = None,
) -> tuple:
    """
    Create train and validation dataloaders.

    Parameters
    ----------
    texts : List[str]
        List of input texts
    tokenizer : Tokenizer
        Trained tokenizer
    batch_size : int, default=16
        Batch size
    max_seq_len : int, default=128
        Maximum sequence length
    train_split : float, default=0.9
        Fraction for training
    dynamic_padding : bool, default=False
        Whether to use dynamic padding
    pad_to_multiple : int, optional
        If provided, pad sequences to multiples of this value

    Returns
    -------
    tuple
        (train_loader, val_loader)
    """
    # Split data
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    # Create datasets
    train_dataset = TextDataset(
        train_texts, tokenizer, max_seq_len, dynamic_padding=dynamic_padding
    )
    val_dataset = TextDataset(
        val_texts, tokenizer, max_seq_len, dynamic_padding=dynamic_padding
    )

    # Create collate function for dynamic padding
    collate_fn = None
    if dynamic_padding:
        pad_token_id = tokenizer.token_to_id("<pad>") or 0
        collate_fn = make_collate_fn(pad_token_id, pad_to_multiple)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader
