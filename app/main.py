import click
from .cli import tokenizer, model


@click.group()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to the YAML configuration file"
)
@click.pass_context
def cli(ctx, config):
    """A lightweight CLI for simple model preprocessing, training, evaluation, and optimization."""
    # Ensure context object exists
    ctx.ensure_object(dict)
    # Store config path in context for subcommands to use
    ctx.obj["config"] = config


# Add commands to the CLI
cli.add_command(tokenizer)
cli.add_command(model)

if __name__ == "__main__":
    cli()
