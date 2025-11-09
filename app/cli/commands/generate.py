import click
import torch
import time
from ..utils import setup_device, load_tokenizer, load_model
from ..utils.config import load_config


@click.command()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to the YAML configuration file"
)
@click.option("--prompt", help="Text prompt to start generation (overrides config)")
@click.option(
    "--prompt-file",
    type=click.Path(exists=True),
    help="Path to a file containing the prompt (alternative to --prompt)",
)
def generate(config, prompt, prompt_file):
    """Generate text using the trained NanoLLM model."""
    try:
        # Load configuration from YAML
        cfg = load_config("generate", config)

        # Handle prompt input from CLI if provided
        if prompt_file:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

        if prompt:
            cfg["prompt"] = prompt

        if not cfg["prompt"]:
            click.echo(
                "Error: Empty prompt provided. Use --prompt or --prompt-file, or set prompt in config.",
                err=True,
            )
            return

        # Display configuration
        click.echo("=== Text Generation Configuration ===")
        click.echo(f"Model path: {cfg['model_path']}")
        click.echo(f"Tokenizer path: {cfg['tokenizer_path']}")
        click.echo(
            f"Prompt: {cfg['prompt'][:100]}{'...' if len(cfg['prompt']) > 100 else ''}"
        )
        click.echo(f"Max new tokens: {cfg['max_new_tokens']}")
        click.echo(f"Temperature: {cfg['temperature']}")
        click.echo(
            f"Sampling: {'Enabled' if cfg['do_sample'] else 'Disabled (greedy)'}"
        )
        click.echo(f"Top-k: {cfg['top_k']}")
        click.echo(f"Top-p: {cfg['top_p']}")
        click.echo(f"Repetition penalty: {cfg['repetition_penalty']}")
        click.echo(f"Number of samples: {cfg['num_samples']}")
        if cfg["output_file"]:
            click.echo(f"Output file: {cfg['output_file']}")
        click.echo("====================================")

        # Set device
        device = setup_device(cfg["device"])

        # Load tokenizer
        tokenizer, vocab_size = load_tokenizer(cfg["tokenizer_path"])
        if tokenizer is None:
            return

        # Load model
        model = load_model(cfg["model_path"], device)
        if model is None:
            return

        model.eval()

        # Tokenize prompt
        encoded = tokenizer.encode(cfg["prompt"])
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)

        # Check sequence length
        max_seq_len = model.max_seq_len
        if input_ids.size(1) >= max_seq_len:
            click.echo(
                f"Warning: Prompt length ({input_ids.size(1)}) exceeds or equals max sequence length ({max_seq_len})",
                err=True,
            )
            click.echo("Truncating prompt to fit...")
            input_ids = input_ids[:, : max_seq_len - 1]

        # Generate text
        click.echo(f"\nGenerating {cfg['num_samples']} sample(s)...")
        start_time = time.time()

        all_outputs = []
        for i in range(cfg["num_samples"]):
            with torch.no_grad():
                # Generate tokens
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=cfg["max_new_tokens"],
                    temperature=cfg["temperature"],
                    do_sample=cfg["do_sample"],
                    top_k=cfg["top_k"],
                    top_p=cfg["top_p"],
                    repetition_penalty=cfg.get("repetition_penalty", 1.0),
                )

                # Decode to text
                generated_text = tokenizer.decode(output_ids[0].tolist())

                # Remove prompt from output if needed
                if not cfg["show_prompt"]:
                    # Find where the generated text starts after the prompt
                    prompt_tokens = tokenizer.encode(cfg["prompt"]).ids
                    generated_tokens = output_ids[0].tolist()

                    # Find the first occurrence of the prompt tokens in the generated tokens
                    if len(prompt_tokens) <= len(generated_tokens):
                        # Simple approach: just skip the prompt length
                        generated_only_tokens = generated_tokens[len(prompt_tokens) :]
                        generated_text = tokenizer.decode(generated_only_tokens)

                all_outputs.append(generated_text)

                # Display progress
                click.echo(f"Sample {i + 1}/{cfg['num_samples']} completed")

        generation_time = time.time() - start_time
        tokens_generated = sum(len(tokenizer.encode(text).ids) for text in all_outputs)
        tokens_per_sec = (
            tokens_generated / generation_time if generation_time > 0 else 0
        )

        # Display results
        click.echo(
            f"\nGeneration completed in {generation_time:.2f}s ({tokens_per_sec:.1f} tokens/sec)"
        )
        click.echo("\n" + "=" * 50)
        click.echo("GENERATION PARAMETERS")
        click.echo("=" * 50)
        click.echo(f"Temperature: {cfg['temperature']}")
        click.echo(f"Top-k: {cfg['top_k']}")
        click.echo(f"Top-p: {cfg['top_p']}")
        click.echo(f"Repetition penalty: {cfg['repetition_penalty']}")
        click.echo(
            f"Sampling: {'Enabled' if cfg['do_sample'] else 'Disabled (greedy)'}"
        )
        click.echo("\n" + "=" * 50)
        click.echo("GENERATED TEXT")
        click.echo("=" * 50)

        for i, text in enumerate(all_outputs):
            click.echo(f"\n--- Sample {i + 1} ---")
            click.echo(text)

        # Save to file if requested
        if cfg["output_file"]:
            with open(cfg["output_file"], "w", encoding="utf-8") as f:
                for i, text in enumerate(all_outputs):
                    f.write(f"--- Sample {i + 1} ---\n")
                    f.write(text)
                    f.write("\n\n")
            click.echo(f"\nGenerated text saved to: {cfg['output_file']}")

    except Exception as e:
        click.echo(f"Error during generation: {e}", err=True)
        import traceback

        click.echo(traceback.format_exc(), err=True)
