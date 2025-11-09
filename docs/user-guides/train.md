# Model Training Command

## Syntax
```bash
python -m app model train --config CONFIG_FILE
```

## Purpose
Trains a transformer model on your tokenized text data.

## Key Configuration
```yaml
train:
  tokenizer_path: "path/to/tokenizer"
  data_files: ["train.txt"]
  total_steps: 1000
  batch_size: 16
  learning_rate: 0.0005
  output_dir: "model_output"
  max_seq_len: 128
  d_model: 256
  n_layers: 6
  n_heads: 8
```

## Training Metrics
- **Perplexity (PPL)**: Lower is better
- **Loss**: Training objective
- **Step/s**: Training speed
- **Memory**: Current memory usage

## Example
```bash
python -m app model train --config configs/small_train.yaml
```

## Tips
- Use `dynamic_padding: true` for efficiency
- Adjust batch size based on GPU memory
- Lower learning rate if training is unstable
- Enable `auto_resume: true` to continue from checkpoints