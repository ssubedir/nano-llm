import os
import yaml
import click
from typing import Dict, Any, Optional


class ConfigLoader:
    """Handles loading and validation of YAML configuration files."""

    DEFAULT_CONFIG_PATH = "config.yaml"

    # Default configuration values
    DEFAULT_CONFIG = {
        "global": {"device": "auto"},
        "tokenizer": {
            "input_files": ["dataset/input.txt"],
            "data_dir": None,
            "output_dir": "tokenizer_output",
            "vocab_size": 10000,
        },
        "train": {
            "tokenizer_path": "tokenizer_output",
            "data_files": ["dataset/input.txt"],
            "data_dir": None,
            "total_steps": 1000,
            "batch_size": 8,
            "learning_rate": 0.001,
            "output_dir": "model_output",
            "max_seq_len": 64,
            "dynamic_padding": True,
            "pad_to_multiple": None,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 512,
            "dropout": 0.1,
            "checkpoint_interval": 250,
            "eval_interval": 100,
            "max_training_hours": None,
            "auto_resume": True,
        },
        "eval": {
            "model_path": "model_output",
            "tokenizer_path": "tokenizer_output",
            "data_file": "dataset/input.txt",
            "batch_size": 16,
            "max_samples": None,
        },
        "generate": {
            "model_path": "model_output",
            "tokenizer_path": "tokenizer_output",
            "prompt": "",
            "prompt_file": None,
            "max_new_tokens": 50,
            "temperature": 0.8,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "output_file": None,
            "num_samples": 1,
            "show_prompt": True,
        },
        "tune.prune": {
            "output_dir": "pruned_model_output",
            "tokenizer_path": "tokenizer_output",
            "device": "auto",
            # Pruning-specific settings
            "model_path": "model_output",
            "method": "magnitude",
            "structured": False,
            "pruning_ratio": 0.5,
            "global_pruning": True,
            "eval_after_pruning": True,
            "save_intermediate": False,
        },
        "tune.distill": {
            "output_dir": "distilled_model_output",
            "tokenizer_path": "tokenizer_output",
            "device": "auto",
            # Distillation-specific settings
            "teacher_model_path": "model_output",
            "student_config": {
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 2,
                "d_ff": 256,
                "dropout": 0.1,
            },
            "temperature": 3.0,
            "alpha": 0.5,
            "beta": 0.5,
            "data_files": ["dataset/input.txt"],
            "data_dir": None,
            "total_steps": 500,
            "batch_size": 8,
            "learning_rate": 0.001,
            "checkpoint_interval": 100,
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigLoader.

        Args:
            config_path: Path to the config file. If None, uses default config.yaml
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        # Start with default configuration
        config = self._deep_copy_dict(self.DEFAULT_CONFIG)

        # Try to load user configuration file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                    click.echo(f"Loaded configuration from {self.config_path}")

                # Merge user config with defaults
                config = self._merge_configs(config, user_config)

            except yaml.YAMLError as e:
                click.echo(f"Error parsing YAML configuration: {e}", err=True)
                click.echo("Using default configuration.", err=True)
            except Exception as e:
                click.echo(f"Error loading configuration file: {e}", err=True)
                click.echo("Using default configuration.", err=True)
        else:
            if self.config_path == self.DEFAULT_CONFIG_PATH:
                # Generate default config.yaml file
                try:
                    with open(self.config_path, "w", encoding="utf-8") as f:
                        yaml.dump(
                            self.DEFAULT_CONFIG, f, default_flow_style=False, indent=2
                        )
                    click.echo(
                        f"No config.yaml found. Generated default configuration file at {self.config_path}"
                    )
                    click.echo("You can edit this file to customize your settings.")
                except Exception as e:
                    click.echo(f"Could not create default config file: {e}", err=True)
                    click.echo("Using default configuration.")
            else:
                click.echo(
                    f"Configuration file '{self.config_path}' not found. Using default configuration.",
                    err=True,
                )

        return config

    def get_config(
        self, command: str, cli_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get configuration for a specific command, with optional CLI overrides.

        Args:
        command: The command name (tokenizer, train, eval, generate, tune)
            cli_overrides: Dictionary of CLI parameters to override config

        Returns:
            Merged configuration dictionary
        """
        if command not in self.config:
            raise ValueError(f"Unknown command: {command}")

        # Start with global settings
        result = self._deep_copy_dict(self.config.get("global", {}))

        # Override with command-specific settings
        result.update(self.config[command])

        # Apply CLI overrides if provided
        if cli_overrides:
            result.update(cli_overrides)

        # Validate the final configuration
        self._validate_config(command, result)

        return result

    def _merge_configs(
        self, default: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge user config with default config."""
        result = self._deep_copy_dict(default)

        for key, value in user.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a dictionary."""
        import copy

        return copy.deepcopy(d)

    def _validate_config(self, command: str, config: Dict[str, Any]) -> None:
        """Validate configuration for a specific command."""
        validation_rules = {
            "tokenizer": {
                "required": ["output_dir"],
                "at_least_one": [["input_files", "data_dir"]],
                "positive_int": ["vocab_size"],
                "file_exists": ["input_files", "data_dir"],
            },
            "train": {
                "required": ["tokenizer_path", "output_dir"],
                "at_least_one": [["data_files", "data_dir"]],
                "positive_int": [
                    "total_steps",
                    "batch_size",
                    "max_seq_len",
                    "d_model",
                    "n_layers",
                    "n_heads",
                    "d_ff",
                    "checkpoint_interval",
                    "eval_interval",
                ],
                "positive_float": ["learning_rate", "max_training_hours"],
                "range_0_1": ["dropout"],
                "file_exists": ["tokenizer_path", "data_files", "data_dir"],
            },
            "eval": {
                "required": ["model_path", "tokenizer_path"],
                "at_least_one": [["data_files", "data_dir"]],
                "positive_int": ["batch_size", "max_samples"],
                "file_exists": [
                    "model_path",
                    "tokenizer_path",
                    "data_files",
                    "data_dir",
                ],
            },
            "generate": {
                "required": ["model_path", "tokenizer_path", "prompt"],
                "positive_int": ["max_new_tokens", "num_samples", "top_k"],
                "positive_float": ["temperature", "top_p"],
                "range_0_1": ["temperature", "top_p"],
                "file_exists": ["model_path", "tokenizer_path"],
            },
            "tune.prune": {
                "required": ["output_dir", "tokenizer_path", "model_path"],
                "range_0_1": ["pruning_ratio", "structured"],
                "file_exists": ["tokenizer_path", "model_path"],
                "one_of": {"method": ["magnitude", "gradient", "random"]},
            },
            "tune.distill": {
                "required": ["output_dir", "tokenizer_path", "teacher_model_path"],
                "at_least_one": [["data_files", "data_dir"]],
                "positive_int": ["total_steps", "batch_size", "checkpoint_interval"],
                "positive_float": ["learning_rate", "temperature"],
                "range_0_1": ["alpha", "beta"],
                "file_exists": [
                    "tokenizer_path",
                    "teacher_model_path",
                    "data_files",
                    "data_dir",
                ],
            },
        }

        if command not in validation_rules:
            return

        rules = validation_rules[command]

        # Check required fields
        for field in rules.get("required", []):
            if field not in config or config[field] is None or config[field] == "":
                # Special case for generate command: prompt can be provided via --prompt-file
                if (
                    command == "generate"
                    and field == "prompt"
                    and config.get("prompt_file")
                ):
                    continue
                raise ValueError(f"Required parameter '{field}' is missing")

        # Check at-least-one rule (must have at least one of the specified fields)
        for field_groups in rules.get("at_least_one", []):
            if not any(config.get(field) for field in field_groups):
                field_names = " or ".join(field_groups)
                raise ValueError(f"At least one of [{field_names}] must be specified")

        # Check positive integers
        for field in rules.get("positive_int", []):
            if field in config and config[field] is not None:
                try:
                    value = int(config[field])
                    if value <= 0:
                        raise ValueError(
                            f"Parameter '{field}' must be a positive integer, got: {value}"
                        )
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Parameter '{field}' must be an integer, got: {config[field]}"
                    )

        # Check positive floats
        for field in rules.get("positive_float", []):
            if field in config and config[field] is not None:
                try:
                    value = float(config[field])
                    if value <= 0:
                        raise ValueError(
                            f"Parameter '{field}' must be a positive number, got: {value}"
                        )
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Parameter '{field}' must be a number, got: {config[field]}"
                    )

        # Check range 0-1
        for field in rules.get("range_0_1", []):
            if field in config and config[field] is not None:
                try:
                    value = float(config[field])
                    if not (0 <= value <= 1):
                        raise ValueError(
                            f"Parameter '{field}' must be between 0 and 1, got: {value}"
                        )
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Parameter '{field}' must be a number, got: {config[field]}"
                    )

        # Check one_of rule (field value must be one of the specified values)
        one_of_rules = rules.get("one_of", {})
        for field, allowed_values in one_of_rules.items():
            if field in config and config[field] is not None:
                if config[field] not in allowed_values:
                    raise ValueError(
                        f"Parameter '{field}' must be one of {allowed_values}, got: {config[field]}"
                    )

        # Check file existence
        for field in rules.get("file_exists", []):
            if field in config and config[field] is not None:
                # Handle list of files (e.g., input_files)
                if isinstance(config[field], list):
                    for file_path in config[field]:
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"File not found: {file_path}")
                else:
                    if not os.path.exists(config[field]):
                        raise FileNotFoundError(
                            f"File or directory not found: {config[field]}"
                        )


def load_config(
    command: str, config_path: Optional[str] = None, **cli_overrides
) -> Dict[str, Any]:
    """
    Convenience function to load configuration for a command.

    Args:
        command: The command name
        config_path: Optional path to config file
        **cli_overrides: CLI parameters to override config

    Returns:
        Merged configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.get_config(command, cli_overrides)
