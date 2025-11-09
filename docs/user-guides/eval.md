# Model Evaluation Command

## Syntax
```bash
python -m app model eval --config CONFIG_FILE
```

## Purpose
Evaluates a trained model on validation data.

## Configuration
```yaml
eval:
  model_path: "path/to/model"
  tokenizer_path: "path/to/tokenizer"
  data_files: ["valid.txt"]
  batch_size: 16
  max_samples: 1000  # null for all samples
```

## Key Metrics
- **Final Perplexity**: Model quality score (lower is better)
- **Tokens/sec**: Processing speed
- **Memory usage**: Current and peak memory

## Example
```bash
python -m app model eval --config configs/small_train.yaml
```

## Results
Evaluation saves results to `{model_path}/evaluation_results.pt` and displays a summary report.

## Tips
- Use data not seen during training
- < 50 PPL is excellent, 50-100 is good, > 200 needs improvement
- Use same tokenizer as training