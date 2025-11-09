import click
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import psutil
from ...data import load_data_sources, create_dataloaders
from ...model import NanoLLM
from ..utils import setup_device, load_tokenizer
from ..utils.config import load_config
from ..utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    TimeTracker,
    format_time_remaining,
    format_elapsed_time,
)


@click.command()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to the YAML configuration file"
)
def train(config):
    """Train the NanoLLM model on the provided data."""
    try:
        # Load configuration from YAML
        cfg = load_config("train", config)

        # Display basic configuration first
        click.echo("=== Model Training Configuration ===")
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

        click.echo(f"Total steps: {cfg['total_steps']}")
        click.echo(f"Batch size: {cfg['batch_size']}")
        click.echo(f"Learning rate: {cfg['learning_rate']}")
        click.echo(f"Output directory: {cfg['output_dir']}")
        click.echo(f"Checkpoint interval: {cfg['checkpoint_interval']} steps")
        click.echo(f"Evaluation interval: {cfg['eval_interval']} steps")

        # Display time-based settings
        if cfg.get("max_training_hours"):
            click.echo(f"Time limit: {cfg['max_training_hours']} hours")
        else:
            click.echo("Time limit: None (unlimited)")
        click.echo(
            f"Auto-resume: {'Enabled' if cfg.get('auto_resume', True) else 'Disabled'}"
        )

        click.echo(
            f"Padding: {'Dynamic' if cfg.get('dynamic_padding', False) else 'Static'}"
        )
        if cfg.get("pad_to_multiple"):
            click.echo(f"Pad to multiple of: {cfg['pad_to_multiple']}")
        click.echo("Model hyperparameters:")
        click.echo(f"  - Max sequence length: {cfg['max_seq_len']}")
        click.echo(f"  - Model dimension: {cfg['d_model']}")
        click.echo(f"  - Layers: {cfg['n_layers']}")
        click.echo(f"  - Heads: {cfg['n_heads']}")
        click.echo(f"  - FF dimension: {cfg['d_ff']}")
        click.echo(f"  - Dropout: {cfg['dropout']}")
        click.echo("======================================")

        # Validate inputs (already done in config validation, but keep for clarity)
        # Note: Validation is now handled by the load_data_sources function

        # Set device
        device = setup_device(cfg["device"])

        # Load tokenizer
        tokenizer, vocab_size = load_tokenizer(cfg["tokenizer_path"])
        if tokenizer is None:
            return

        # Ensure tokenizer was loaded successfully
        if tokenizer is None:
            click.echo("Error: Failed to initialize tokenizer", err=True)
            return

        # Load data using the new load_data_sources function
        try:
            texts = load_data_sources(
                data_files=cfg.get("data_files"), data_dir=cfg.get("data_dir")
            )
            click.echo(f"Loaded {len(texts)} texts from data sources")
        except Exception as e:
            click.echo(f"Error loading data: {e}", err=True)
            return

        # Create dataloaders with dynamic padding support
        train_loader, val_loader = create_dataloaders(
            texts,
            tokenizer,
            batch_size=cfg["batch_size"],
            max_seq_len=cfg["max_seq_len"],
            dynamic_padding=cfg.get("dynamic_padding", False),
            pad_to_multiple=cfg.get("pad_to_multiple"),
        )
        click.echo(f"Created dataloaders with {len(train_loader)} training batches")

        # Initialize model
        model = NanoLLM(
            vocab_size=vocab_size,
            max_seq_len=cfg["max_seq_len"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
        ).to(device)

        click.echo(
            f"Initialized NanoLLM model with {sum(p.numel() for p in model.parameters())} parameters"
        )

        # Setup optimizer and loss
        optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token

        # Create output directory
        os.makedirs(cfg["output_dir"], exist_ok=True)

        # Initialize time tracker
        time_tracker = TimeTracker(max_hours=cfg.get("max_training_hours"))

        # Check for resume functionality
        start_step = 0
        if cfg.get("auto_resume", True):
            latest_checkpoint = find_latest_checkpoint(cfg["output_dir"])
            if latest_checkpoint:
                checkpoint = load_checkpoint(latest_checkpoint, device)
                if checkpoint:
                    # Determine if we should resume
                    should_resume = True
                    if checkpoint.get("checkpoint_type") == "time_limit":
                        click.echo("\n=== Training Resume Information ===")
                        click.echo(
                            f"Found time-limit checkpoint from step {checkpoint['step']}"
                        )

                        elapsed = checkpoint.get("training_metadata", {}).get(
                            "total_elapsed_time", 0
                        )
                        click.echo(
                            f"Total elapsed time: {format_elapsed_time(elapsed)}"
                        )
                        click.echo(
                            "Note: Continuing training until step target is reached..."
                        )
                        click.echo("====================================\n")

                        # Reset time tracking to allow continuation to step target
                        time_tracker.initialize(reset_elapsed=True)
                        should_resume = True
                    else:
                        click.echo(f"Found checkpoint from step {checkpoint['step']}")
                        if click.confirm("Resume from this checkpoint?"):
                            click.echo("Resuming training...")
                        else:
                            click.echo("Starting fresh training...")
                            should_resume = False

                    if should_resume:
                        model.load_state_dict(checkpoint["model_state_dict"])
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        start_step = checkpoint["step"] + 1

                        # Time tracker already initialized above for time-limit checkpoints
                        pass

        # Initialize time tracker if not already done
        if not time_tracker.initialized:
            time_tracker.initialize()

        # Training loop
        click.echo("Starting step-based training...")
        model.train()

        # Create infinite data iterator for step-based training
        data_iter = iter(train_loader)

        # Training metrics
        start_time = time.time()
        step_times = []

        for step in range(start_step, cfg["total_steps"]):
            step_start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart iteration if we run out of data
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Prepare targets (shift input_ids for next token prediction)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            # Calculate loss
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            # Backward pass
            loss.backward()

            # Calculate gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Calculate metrics
            step_time = time.time() - step_start
            step_times.append(step_time)

            # Calculate moving average of step time
            if len(step_times) > 10:
                step_times.pop(0)
            avg_step_time = sum(step_times) / len(step_times)
            steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

            # Calculate perplexity
            perplexity = math.exp(loss.item())

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Get memory usage
            if device == "cuda":
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

            # Format elapsed time
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Update time tracker
            time_tracker.update()

            # Check time limit
            if time_tracker.is_time_limit_reached():
                # Save time-limit checkpoint
                checkpoint_config = {
                    "vocab_size": vocab_size,
                    "max_seq_len": cfg["max_seq_len"],
                    "d_model": cfg["d_model"],
                    "n_layers": cfg["n_layers"],
                    "n_heads": cfg["n_heads"],
                    "d_ff": cfg["d_ff"],
                    "dropout": cfg["dropout"],
                }
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    loss,
                    checkpoint_config,
                    cfg["output_dir"],
                    checkpoint_type="time_limit",
                    training_metadata=time_tracker.get_metadata(),
                )
                click.echo(
                    f"\nTime limit of {cfg.get('max_training_hours')} hours reached."
                )
                click.echo(
                    f"Training stopped at step {step}. Total training time: {format_elapsed_time(time_tracker.total_elapsed_time)}"
                )
                click.echo("Resume by running the same command again.")
                return

            # Get remaining time for display
            remaining_seconds = time_tracker.get_remaining_time()
            remaining_str = (
                format_time_remaining(remaining_seconds)
                if remaining_seconds != float("inf")
                else "∞"
            )

            # Log progress with detailed metrics - PPL highlighted as most important
            if step % 50 == 0:
                click.echo(
                    f"Step {step:4d} | PERPLEXITY (PPL): {perplexity:8.2f} | Loss: {loss.item():7.4f} | LR: {current_lr:.2e} | "
                    f"Step/s: {steps_per_sec:4.1f} | Time: {time_str} | Remaining: {remaining_str} | "
                    f"Grad norm: {grad_norm:.4f} | Memory: {memory_mb:.1f}MB"
                )

            # Save checkpoint
            if step > 0 and step % cfg["checkpoint_interval"] == 0:
                checkpoint_config = {
                    "vocab_size": vocab_size,
                    "max_seq_len": cfg["max_seq_len"],
                    "d_model": cfg["d_model"],
                    "n_layers": cfg["n_layers"],
                    "n_heads": cfg["n_heads"],
                    "d_ff": cfg["d_ff"],
                    "dropout": cfg["dropout"],
                }
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    loss,
                    checkpoint_config,
                    cfg["output_dir"],
                    checkpoint_type="regular",
                    training_metadata=time_tracker.get_metadata(),
                )
                click.echo(f"Checkpoint saved at step {step}")

        # Update time tracker one final time
        time_tracker.update()

        # Save final model
        final_checkpoint = {
            "step": cfg["total_steps"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "max_seq_len": cfg["max_seq_len"],
                "d_model": cfg["d_model"],
                "n_layers": cfg["n_layers"],
                "n_heads": cfg["n_heads"],
                "d_ff": cfg["d_ff"],
                "dropout": cfg["dropout"],
            },
            "timestamp": time.time(),
            "checkpoint_type": "final",
            "training_metadata": time_tracker.get_metadata(),
        }
        torch.save(final_checkpoint, os.path.join(cfg["output_dir"], "model_final.pt"))
        click.echo(f"\nTraining completed after {cfg['total_steps']} steps!")
        click.echo(
            f"Total training time: {format_elapsed_time(time_tracker.total_elapsed_time)}"
        )
        click.echo(f"Model saved to {cfg['output_dir']}")

    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
