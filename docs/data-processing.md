# Data Processing Documentation

This document details the data processing pipeline in nano-llm, including text preprocessing, tokenization, dataset creation, and data loading strategies.

## Overview

The data processing pipeline transforms raw text into format suitable for model training and evaluation:

```
Raw Text → Data Loading → Text Cleaning → Tokenization → Dataset Creation → DataLoader → Model
```

## Data Sources

nano-llm supports multiple ways to specify data sources:

### Individual Files
```yaml
train:
  data_files: ["file1.txt", "file2.txt", "file3.txt"]
```

### Directory
```yaml
train:
  data_dir: "path/to/directory"
```
All `.txt` files in the directory and its subdirectories will be included.

### Data Loading Process

```python
def load_data_sources(data_files=None, data_dir=None):
    """
    Load text data from files or directories
    
    Args:
        data_files: List of specific file paths
        data_dir: Directory path to scan for .txt files
    
    Returns:
        List of text strings, one per line/document
    """
```

### File Format Support

**Aany Plain Text Files (.txt, .md)....**:
- UTF-8 encoding is preferred
- Each line can be treated as a separate document
- Paragraph breaks can be preserved or normalized

## Text Preprocessing

### Cleaning Operations

The default preprocessing includes:

1. **Encoding Handling**: UTF-8 encoding with fallback to latin-1
2. **Whitespace Normalization**: Standardize line endings and spaces
3. **Empty Line Filtering**: Remove completely empty lines
4. **Length Filtering**: Optionally filter very short/long documents

## Tokenization

### Byte-Pair Encoding (BPE)

nano-llm uses BPE tokenization which:
- Handles out-of-vocabulary words efficiently
- Maintains a fixed vocabulary size
- Preserves subword information

### Tokenizer Training

```python
class TokenizerTrainer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.tokenizer = None
    
    def train_tokenizer(self, texts, output_dir):
        """
        Train BPE tokenizer on provided texts
        
        Args:
            texts: List of text strings
            output_dir: Directory to save tokenizer
            
        Returns:
            Trained tokenizer object
        """
```

### Tokenizer Output

Training a tokenizer creates multiple files:

```
tokenizer_output/
├── tokenizer.json          # Main tokenizer definition (HuggingFace format)
├── config.json             # Tokenizer configuration
├── merges.txt              # BPE merge rules
├── vocab.json              # Vocabulary with token IDs
└── special_tokens_map.txt  # Special token mappings
```

### Using Trained Tokenizer

```python
# Loading tokenizer
tokenizer = load_tokenizer("path/to/tokenizer_output")

# Encoding text
encoded = tokenizer.encode("Sample text")
tokens = encoded.ids  # List of token IDs

# Decoding tokens
text = tokenizer.decode(token_ids)
```

## Dataset Creation

### TextDataset Class

The `TextDataset` class handles the conversion from text to tensors:

```python
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len):
        """
        Create dataset from texts and tokenizer
        
        Args:
            texts: List of text strings
            tokenizer: Trained tokenizer
            max_seq_len: Maximum sequence length
        """
```

### Dataset Processing

1. **Tokenization**: Convert each text to token IDs
2. **Truncation**: Cut sequences to max_seq_len
3. **Padding**: Add padding tokens to reach consistent length
4. **Attention Mask**: Create mask to distinguish real tokens from padding

### Example Processing

```
Input: "The cat sat on the mat"
Tokens: [12, 45, 78, 23, 12, 89]
With padding: [12, 45, 78, 23, 12, 89, 0, 0]
Attention mask: [1, 1, 1, 1, 1, 1, 0, 0]
```

## Batching and Padding

### Static Padding

Traditional approach with fixed-length sequences:
- All sequences padded to the same length
- Simple but computationally wasteful
- Used when memory is abundant

### Dynamic Padding

nano-llm implements dynamic padding for efficiency:

```python
def dynamic_padding_collate(batch):
    """
    Collate function with dynamic padding
    
    Each batch is padded to the maximum length in that batch,
    not the global maximum length.
    """
```

**Benefits:**
- Less computation on padding tokens
- Faster training speeds
- Better memory utilization

### Padding to Multiples

For GPU efficiency, sequences can be padded to multiples of 8 or 16:

```yaml
train:
  dynamic_padding: true
  pad_to_multiple: 8  # Pad each batch to length % 8 == 0
```

## DataLoader Creation

### create_dataloaders Function

```python
def create_dataloaders(texts, tokenizer, batch_size, max_seq_len,
                       dynamic_padding=False, pad_to_multiple=None,
                       train_split=0.9, random_seed=42):
    """
    Create training and validation dataloaders
    
    Args:
        texts: List of text strings
        tokenizer: Trained tokenizer
        batch_size: Batch size for dataloaders
        max_seq_len: Maximum sequence length
        dynamic_padding: Enable dynamic padding
        pad_to_multiple: Pad to multiples of N
        train_split: Fraction of data for training
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader
    """
```

### DataLoader Features

1. **Automatic Shuffling**: Training data is shuffled each epoch
2. **Pin Memory**: Faster CPU-to-GPU transfer when using GPU
3. **Drop Last**: Drop incomplete final batch for consistent batch sizes
4. **Persistent Workers**: Keep workers alive between epochs

## Next Steps

- Check the [Configuration Reference](configuration.md) for data-related settings