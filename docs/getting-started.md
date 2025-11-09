# Getting Started

This tutorial walks you through training a tokenizer, model, evaluation, and text generation.

## 1. Train a Tokenizer

```bash
python -m app tokenizer --config configs/small_train.yaml
```

This creates a tokenizer with 5,000 vocabulary size in `small_tokenizer_output/`.

## 2. Train a Model

```bash
python -m app model train --config configs/small_train.yaml
```

Training metrics:
- **Perplexity (PPL)**: Lower is better
- **Loss**: Training objective
- **Step/s**: Training speed

## 3. Evaluate the Model

```bash
python -m app model eval --config configs/small_train.yaml
```

Evaluation shows:
- Final perplexity score
- Performance metrics
- Memory usage

## 4. Generate Text

```bash
python -m app model generate --config configs/small_train.yaml
```

You can override the prompt:
```bash
python -m app model generate --config configs/small_train.yaml --prompt "Your custom prompt"
```

## Configuration Example

```yaml
train:
  tokenizer_path: "small_tokenizer_output"
  data_files: ["dataset/wikitext-103-raw-v1/wikitext-103-raw/wiki.train.raw"]
  total_steps: 100
  batch_size: 4
  learning_rate: 0.001
  output_dir: "small_model_output"
  max_seq_len: 64
  d_model: 64
  n_layers: 1
  n_heads: 2
  d_ff: 256
  dropout: 0.1
```

## Next Steps

- Try the `big_train.yaml` configuration for larger models
- Modify configurations to experiment with different settings
- Use your own text data