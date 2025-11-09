import os
import torch
import click
from ...model import NanoLLM


def setup_device(device_str):
    """Configure and return the appropriate device."""
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str
    click.echo(f"Using device: {device}")
    return device


def load_tokenizer(tokenizer_path):
    """Load tokenizer with error handling."""
    # Validate tokenizer path
    if not os.path.exists(tokenizer_path):
        click.echo(
            f"Error: Tokenizer path '{tokenizer_path}' does not exist.", err=True
        )
        return None, None

    # Check for required files
    vocab_path = os.path.join(tokenizer_path, "vocab.json")
    merges_path = os.path.join(tokenizer_path, "merges.txt")

    if not os.path.exists(vocab_path):
        click.echo(f"Error: Tokenizer vocab file not found at {vocab_path}", err=True)
        return None, None

    if not os.path.exists(merges_path):
        click.echo(f"Error: Tokenizer merges file not found at {merges_path}", err=True)
        return None, None

    tokenizer = None
    vocab_size = None
    try:
        # Check if vocab.json is a valid JSON file
        import json

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            click.echo(f"Vocab file is a valid JSON with {len(vocab_data)} entries")

        # Try loading with the correct method
        from tokenizers.implementations import ByteLevelBPETokenizer

        tokenizer = ByteLevelBPETokenizer.from_file(vocab_path, merges_path)
        vocab_size = tokenizer.get_vocab_size()
        click.echo(f"Loaded tokenizer from {tokenizer_path}")
        click.echo(f"Detected vocabulary size: {vocab_size}")
    except json.JSONDecodeError as e:
        click.echo(f"Error: vocab.json is not valid JSON: {e}", err=True)
        click.echo(f"File path: {vocab_path}", err=True)
        with open(vocab_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            click.echo(f"First line of vocab.json: {first_line}", err=True)
        return None, None
    except Exception as e:
        click.echo(f"Error loading tokenizer: {e}", err=True)
        click.echo(
            "Please ensure you have trained the tokenizer first using: uv run -m app.main tokenizer",
            err=True,
        )
        return None, None

    return tokenizer, vocab_size


def load_model(model_path, device):
    """Load model from checkpoint with fallback logic."""
    # Validate model path
    if not os.path.exists(model_path):
        click.echo(f"Error: Model path '{model_path}' does not exist.", err=True)
        return None

    # Try to load model
    checkpoint_path = os.path.join(model_path, "model_final.pt")
    model = None

    if not os.path.exists(checkpoint_path):
        # Look for any checkpoint file
        checkpoint_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
        checkpoint_files.sort()

        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(model_path, checkpoint_file)
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if "config" in checkpoint:
                    config = checkpoint["config"]
                    model = NanoLLM(**config).to(device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    click.echo(f"Loaded model from {checkpoint_file}")
                    break
            except Exception:
                continue
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        model = NanoLLM(**config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        click.echo("Loaded model from model_final.pt")

    if model is None:
        click.echo("Error: Could not load model from any checkpoint.", err=True)
        return None

    model.eval()
    return model
