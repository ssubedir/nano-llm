# Command Guides

This directory contains detailed documentation for all nano-llm commands.

## Available Guides

- [Tokenizer Guide](tokenizer.md) - Training and using custom tokenizers
- [Training Guide](train.md) - Training transformer-based language models
- [Evaluation Guide](eval.md) - Evaluating models with perplexity metrics
- [Generation Guide](generate.md) - Generating text using trained models

## Command Overview

nano-llm provides a unified command-line interface for:

1. **Tokenizer Operations**
   ```bash
   python -m app tokenizer --config configs/small_train.yaml
   ```

2. **Model Training**
   ```bash
   python -m app model train --config configs/small_train.yaml
   ```

3. **Model Evaluation**
   ```bash
   python -m app model eval --config configs/small_train.yaml
   ```

4. **Text Generation**
   ```bash
   python -m app model generate --config configs/small_train.yaml
   ```

For a complete workflow example, see the [Getting Started Guide](../getting-started.md).