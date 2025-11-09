# Configuration Reference

Configuration files use YAML format with sections for different components.

## Global Settings
```yaml
global:
  device: "auto"  # auto, cpu, cuda, mps
```

## Tokenizer Configuration
```yaml
tokenizer:
  input_files: ["file1.txt", "file2.txt"]  # or data_dir
  output_dir: "tokenizer_output"
  vocab_size: 10000
```

## Training Configuration
```yaml
train:
  tokenizer_path: "path/to/tokenizer"
  data_files: ["train.txt"]  # or data_dir
  total_steps: 1000
  batch_size: 16
  learning_rate: 0.0005
  output_dir: "model_output"
  max_seq_len: 128
  d_model: 256
  n_layers: 6
  n_heads: 8
  d_ff: 1024
  dropout: 0.1
  checkpoint_interval: 100
  eval_interval: 100
  dynamic_padding: true
  auto_resume: true
```

## Evaluation Configuration
```yaml
eval:
  model_path: "path/to/model"
  tokenizer_path: "path/to/tokenizer"
  data_files: ["valid.txt"]  # or data_dir
  batch_size: 16
  max_samples: 1000  # null for all samples
```

## Generation Configuration
```yaml
generate:
  model_path: "path/to/model"
  tokenizer_path: "path/to/tokenizer"
  prompt: "Your prompt here"
  max_new_tokens: 100
  temperature: 0.8
  do_sample: true
  top_k: 50
  top_p: 0.9
  repetition_penalty: 1.2
  num_samples: 3
  show_prompt: true
```

## Key Parameters

- **device**: Hardware to use
- **vocab_size**: Size of tokenizer vocabulary
- **total_steps**: Number of training steps
- **batch_size**: Sequences per batch
- **learning_rate**: Training learning rate
- **max_seq_len**: Maximum sequence length
- **d_model**: Model dimension
- **n_layers**: Number of transformer layers
- **n_heads**: Number of attention heads
- **temperature**: Generation randomness (0.1-2.0)