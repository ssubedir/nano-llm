import click
from .commands import train, eval, generate


@click.group()
def model():
    """Model related commands."""
    pass


# Register subcommands
model.add_command(train)
model.add_command(eval)
model.add_command(generate)
