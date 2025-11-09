# nano-llm

A lightweight CLI for simple model preprocessing, training, evaluation, and optimization.

## Overview

nano-llm is a streamlined command-line interface for simple transformer model development. It provides a complete workflow from data preprocessing to model training, evaluation, and text generation, designed for educational purposes and small-scale experiments.

### Key Features

- **Tokenizer Training**: Train custom tokenizers on your text data
- **Model Training**: Train transformer-based language models
- **Model Evaluation**: Evaluate models with perplexity metrics
- **Text Generation**: Generate text using trained models
- **YAML Configuration**: Simple configuration-based approach

### TODO Features (Planned Enhancements)

- **Model Optimization**: Implement pruning and distillation techniques
- **Advanced Positional Encoding**: Add Rotary positional encoding (RoPE) for better performance
- **Efficient Normalization**: Replace LayerNorm with RMSNorm for improved efficiency
- **HuggingFace Integration**: Use the transformers package to enable model deployment to HuggingFace Hub

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ssubedir/nano-llm.git
cd nano-llm

# Install dependencies
uv sync
```

### Basic Usage

For a complete workflow example, see the [Getting Started Guide](docs/getting-started.md).


## Documentation

For detailed information about commands, configuration, and examples, see the [docs](docs/) folder:

- [Getting Started](docs/getting-started.md) - Complete workflow tutorial
- [Configuration Reference](docs/configuration.md) - All configuration options
- [Command Guides](docs/user-guides/README.md) - Detailed command documentation:
  - [Tokenizer](docs/user-guides/tokenizer.md)
  - [Training](docs/user-guides/train.md)
  - [Evaluation](docs/user-guides/eval.md)
  - [Generation](docs/user-guides/generate.md)

## Project Structure

```
nano-llm/
├── app/                   # Core application code
│   ├── cli/               # Command-line interface
│   ├── data/              # Data processing
│   └── model/             # Model architecture
├── configs/               # Configuration files
├── dataset/               # Sample datasets
└── docs/                  # Documentation
```

## Requirements

- Python 3.12+
- PyTorch
- CUDA-compatible GPU (recommended for training)

## Development

```bash
# Lint all files in the current directory
uvx ruff check

# Format all files in the current directory
uvx ruff format
```

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run linting and formatting:
   ```bash
   uvx ruff check
   uvx ruff format
   ```
5. Submit a pull request

See the [TODO section](#todo-features-planned-enhancements) for areas that need work.

## License

MIT License - see the [LICENSE](LICENSE) file for details.
