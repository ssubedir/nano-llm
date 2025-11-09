import os
import torch
import time
import click
from typing import Dict, Any, Optional


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the output directory.

    Args:
        output_dir: Directory containing checkpoints

    Returns:
        Path to the latest checkpoint file, or None if none found
    """
    if not os.path.exists(output_dir):
        return None

    checkpoint_files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]

    if not checkpoint_files:
        return None

    # Sort by step number
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    return os.path.join(output_dir, checkpoint_files[-1])


def load_checkpoint(checkpoint_path: str, device: str) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint with error handling.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors on

    Returns:
        Checkpoint dictionary or None if loading failed
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint
    except Exception as e:
        click.echo(f"Error loading checkpoint {checkpoint_path}: {e}", err=True)
        return None


def save_checkpoint(
    model,
    optimizer,
    step: int,
    loss: float,
    config: Dict[str, Any],
    output_dir: str,
    checkpoint_type: str = "regular",
    training_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save training checkpoint with enhanced metadata.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        step: Current training step
        loss: Current loss value
        config: Model configuration
        output_dir: Directory to save checkpoint
        checkpoint_type: Type of checkpoint ("regular" or "time_limit")
        training_metadata: Additional training metadata

    Returns:
        Path to saved checkpoint
    """
    current_time = time.time()

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item() if hasattr(loss, "item") else loss,
        "config": {
            "vocab_size": config["vocab_size"],
            "max_seq_len": config["max_seq_len"],
            "d_model": config["d_model"],
            "n_layers": config["n_layers"],
            "n_heads": config["n_heads"],
            "d_ff": config["d_ff"],
            "dropout": config["dropout"],
        },
        "timestamp": current_time,
        "checkpoint_type": checkpoint_type,
        "training_metadata": training_metadata or {},
    }

    if checkpoint_type == "time_limit":
        filename = f"checkpoint_time_limit_step_{step}.pt"
        click.echo(
            f"\n⏰ Time limit reached. Saving time-limit checkpoint at step {step}"
        )
    else:
        filename = f"checkpoint_step_{step}.pt"

    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def format_time_remaining(seconds: float) -> str:
    """
    Format remaining time as HH:MM:SS.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds <= 0:
        return "00:00:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_elapsed_time(seconds: float) -> str:
    """
    Format elapsed time in a human-readable way.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class TimeTracker:
    """
    Tracks training time with support for resume functionality.
    """

    def __init__(self, max_hours: Optional[float] = None):
        """
        Initialize time tracker.

        Args:
            max_hours: Maximum training duration in hours
        """
        self.max_hours = max_hours
        self.start_time = None
        self.total_elapsed_time = 0.0
        self.last_save_time = None
        self.initialized = False

    def initialize(
        self,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
        reset_elapsed: bool = False,
    ):
        """
        Initialize the time tracker, potentially from checkpoint metadata.

        Args:
            checkpoint_metadata: Metadata from loaded checkpoint
            reset_elapsed: Whether to reset elapsed time to 0
        """
        self.start_time = time.time()
        self.last_save_time = self.start_time

        if checkpoint_metadata and not reset_elapsed:
            self.total_elapsed_time = checkpoint_metadata.get("total_elapsed_time", 0.0)
            click.echo(
                f"Resuming with {format_elapsed_time(self.total_elapsed_time)} of previous training"
            )
        else:
            self.total_elapsed_time = 0.0
            click.echo("Starting fresh training")

        self.initialized = True

        if self.max_hours:
            remaining_time = self.get_remaining_time()
            click.echo(
                f"Time limit: {self.max_hours:.1f} hours, remaining: {format_time_remaining(remaining_time)}"
            )

    def update(self) -> float:
        """
        Update elapsed time since last update.

        Returns:
            Current total elapsed time
        """
        if not self.initialized:
            return 0.0

        current_time = time.time()
        session_time = current_time - self.last_save_time
        self.total_elapsed_time += session_time
        self.last_save_time = current_time

        return self.total_elapsed_time

    def get_remaining_time(self) -> float:
        """
        Get remaining time in seconds.

        Returns:
            Remaining seconds (0 if no limit or time exceeded)
        """
        if not self.max_hours or not self.initialized:
            return float("inf")

        max_seconds = self.max_hours * 3600
        remaining = max_seconds - self.total_elapsed_time

        # Add a small buffer to allow at least one more step when resuming
        if remaining < 10 and remaining > 0:  # Less than 10 seconds but not zero
            remaining = 10  # Give 10 seconds buffer

        return max(0, remaining)

    def is_time_limit_reached(self) -> bool:
        """
        Check if time limit has been reached.

        Returns:
            True if time limit reached or exceeded
        """
        if not self.max_hours or not self.initialized:
            return False

        return self.get_remaining_time() <= 0

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get time tracking metadata for checkpoint saving.

        Returns:
            Dictionary with time tracking metadata
        """
        return {
            "total_elapsed_time": self.total_elapsed_time,
            "max_hours": self.max_hours,
            "start_time": self.start_time,
            "last_save_time": self.last_save_time,
        }
