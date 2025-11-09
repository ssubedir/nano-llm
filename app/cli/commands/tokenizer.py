import click
import os
from ...data import TokenizerTrainer, load_data_sources
from ..utils.config import load_config


@click.command()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to the YAML configuration file"
)
def tokenizer(config):
    """Train a tokenizer on the provided text data."""
    try:
        # Load configuration from YAML
        cfg = load_config("tokenizer", config)

        # Display current configuration
        click.echo("=== Tokenizer Configuration ===")

        # Display data source information (prioritize directory over files)
        if cfg.get("data_dir"):
            click.echo(f"Data directory: {cfg['data_dir']}")
        elif cfg.get("input_files"):
            if isinstance(cfg["input_files"], list):
                click.echo(f"Input files: {len(cfg['input_files'])} files")
                for file in cfg["input_files"][:3]:  # Show first 3 files
                    click.echo(f"  - {file}")
                if len(cfg["input_files"]) > 3:
                    click.echo(f"  ... and {len(cfg['input_files']) - 3} more files")
            else:
                click.echo(f"Input files: {cfg['input_files']}")

        click.echo(f"Output directory: {cfg['output_dir']}")
        click.echo(f"Vocabulary size: {cfg['vocab_size']}")
        click.echo("===============================")

        # Validate vocab size is positive integer (already done in config validation)
        if not isinstance(cfg["vocab_size"], int) or cfg["vocab_size"] <= 0:
            click.echo(
                f"Error: Vocab size must be a positive integer. Got: {cfg['vocab_size']}",
                err=True,
            )
            return

        # Create output directory if it doesn't exist
        os.makedirs(cfg["output_dir"], exist_ok=True)

        # Load text data using the new load_data_sources function
        try:
            texts = load_data_sources(
                data_files=cfg.get("input_files"), data_dir=cfg.get("data_dir")
            )
            click.echo(f"Loaded {len(texts)} texts from data sources")
        except Exception as e:
            click.echo(f"Error loading data: {e}", err=True)
            return

        # Train tokenizer
        trainer = TokenizerTrainer(vocab_size=cfg["vocab_size"])
        trainer.train_tokenizer(texts, cfg["output_dir"])
        click.echo(f"Tokenizer trained and saved to {cfg['output_dir']}")

    except Exception as e:
        click.echo(f"Error training tokenizer: {e}", err=True)
