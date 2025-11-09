import click
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import math
import psutil
from ...data import load_data_sources, TextDataset
from ..utils import setup_device, load_tokenizer, load_model
from ..utils.config import load_config


@click.command()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to the YAML configuration file"
)
def eval(config):
    """Evaluate the trained NanoLLM model with comprehensive metrics."""
    try:
        # Load configuration from YAML
        cfg = load_config("eval", config)

        # Display configuration
        click.echo("=== Model Evaluation Configuration ===")
        click.echo(f"Model path: {cfg['model_path']}")
        click.echo(f"Tokenizer path: {cfg['tokenizer_path']}")

        # Display data source information (prioritize directory over files)
        if cfg.get("data_dir"):
            click.echo(f"Data directory: {cfg['data_dir']}")
        elif cfg.get("data_files"):
            if isinstance(cfg["data_files"], list):
                click.echo(f"Data files: {len(cfg['data_files'])} files")
                for file in cfg["data_files"][:3]:  # Show first 3 files
                    click.echo(f"  - {file}")
                if len(cfg["data_files"]) > 3:
                    click.echo(f"  ... and {len(cfg['data_files']) - 3} more files")
            else:
                click.echo(f"Data files: {cfg['data_files']}")

        click.echo(f"Batch size: {cfg['batch_size']}")
        if cfg["max_samples"]:
            click.echo(f"Max samples: {cfg['max_samples']}")
        click.echo("======================================")

        # Set device
        device = setup_device(cfg["device"])

        # Load tokenizer
        tokenizer, vocab_size = load_tokenizer(cfg["tokenizer_path"])
        if tokenizer is None:
            return

        # Load and limit data if specified using the new load_data_sources function
        try:
            texts = load_data_sources(
                data_files=cfg.get("data_files"), data_dir=cfg.get("data_dir")
            )
            click.echo(f"Loaded {len(texts)} texts from data sources")

            if cfg["max_samples"] and len(texts) > cfg["max_samples"]:
                texts = texts[: cfg["max_samples"]]
                click.echo(f"Limited to {len(texts)} samples for evaluation")
        except Exception as e:
            click.echo(f"Error loading data: {e}", err=True)
            return

        # Load model first to get the correct max_seq_len
        model = load_model(cfg["model_path"], device)
        if model is None:
            return

        # Create evaluation dataloader (use max_seq_len from loaded model)
        model_max_seq_len = model.max_seq_len
        val_dataset = TextDataset(texts, tokenizer, max_seq_len=model_max_seq_len)
        val_loader = DataLoader(
            val_dataset, batch_size=cfg["batch_size"], shuffle=False
        )
        click.echo(
            f"Created evaluation dataloader with {len(val_loader)} batches (max_seq_len={model_max_seq_len})"
        )

        # Evaluation metrics
        click.echo("Starting evaluation...")
        start_time = time.time()

        total_loss = 0
        total_tokens = 0
        num_batches = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        losses = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Prepare targets
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                attention_mask = attention_mask[:, :-1].contiguous()

                # Forward pass
                logits = model(input_ids, attention_mask)

                # Calculate loss
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                batch_loss = loss.item()

                total_loss += batch_loss
                total_tokens += targets.numel()
                num_batches += 1
                losses.append(batch_loss)

                # Progress update with running perplexity
                if batch_idx % 10 == 0 or batch_idx == len(val_loader) - 1:
                    progress = (batch_idx + 1) / len(val_loader) * 100
                    elapsed = time.time() - start_time
                    current_ppl = math.exp(batch_loss)
                    click.echo(
                        f"Evaluating: {progress:.1f}% ({batch_idx + 1}/{len(val_loader)}) - "
                        f"PERPLEXITY (PPL): {current_ppl:.2f} - Loss: {batch_loss:.4f} - Elapsed: {elapsed:.1f}s"
                    )

        # Calculate final metrics
        eval_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)

        # Loss statistics
        loss_std = torch.tensor(losses).std().item() if len(losses) > 1 else 0
        loss_min = min(losses) if len(losses) > 0 else 0
        loss_max = max(losses) if len(losses) > 0 else 0

        # Token statistics
        tokens_per_sec = total_tokens / eval_time if eval_time > 0 else 0

        # Memory usage
        if device == "cuda":
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            max_memory_mb = memory_mb

        # Model size
        model_params = sum(p.numel() for p in model.parameters())
        model_size_mb = model_params * 4 / 1024 / 1024  # Assuming float32

        # Format evaluation time
        hours = int(eval_time // 3600)
        minutes = int((eval_time % 3600) // 60)
        seconds = int(eval_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        click.echo("\n" + "=" * 50)
        click.echo("EVALUATION RESULTS")
        click.echo("=" * 50)
        click.echo(f"Dataset: {len(texts)} samples, {total_tokens:,} tokens")
        click.echo(f"Model: {model_params:,} parameters ({model_size_mb:.1f}MB)")
        click.echo(f"Device: {device.upper()}")
        click.echo(f"Evaluation time: {time_str} ({eval_time:.1f}s)")
        click.echo("-" * 50)
        click.echo("METRICS:")
        click.echo(f"  FINAL PERPLEXITY (PPL): {perplexity:.2f}")
        click.echo(f"  Average Loss:     {avg_loss:.4f} ± {loss_std:.4f}")
        click.echo(f"  Loss Range:      {loss_min:.4f} / {loss_max:.4f}")
        click.echo("-" * 50)
        click.echo("Performance:")
        click.echo(f"  Tokens/sec: {tokens_per_sec:.1f}")
        click.echo(f"  Batch size: {cfg['batch_size']}")
        click.echo(f"  Throughput: {len(val_loader) / eval_time:.2f} batches/sec")
        click.echo("-" * 50)
        click.echo("Memory usage:")
        click.echo(f"  Current: {memory_mb:.1f}MB")
        click.echo(f"  Peak: {max_memory_mb:.1f}MB")
        click.echo("=" * 50)

        # Save evaluation results
        results = {
            "avg_loss": avg_loss,
            "perplexity": perplexity,
            "tokens_evaluated": total_tokens,
            "samples_evaluated": len(texts),
            "num_batches": num_batches,
            "model_params": model_params,
            "tokens_per_sec": tokens_per_sec,
            "eval_time": eval_time,
        }

        results_path = os.path.join(cfg["model_path"], "evaluation_results.pt")
        torch.save(results, results_path)
        click.echo(f"\nEvaluation results saved to: {results_path}")

    except Exception as e:
        click.echo(f"Error during evaluation: {e}", err=True)
