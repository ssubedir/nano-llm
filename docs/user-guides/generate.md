# Text Generation Command

## Syntax
```bash
python -m app model generate --config CONFIG_FILE [--prompt "TEXT"] [--prompt-file PATH]
```

## Purpose
Generates text using a trained model.

## Configuration
```yaml
generate:
  model_path: "path/to/model"
  tokenizer_path: "path/to/tokenizer"
  prompt: "Default prompt"
  max_new_tokens: 100
  temperature: 0.8
  do_sample: true
  top_k: 50
  top_p: 0.9
  repetition_penalty: 1.2
  num_samples: 3
```

## Parameters
- **temperature**: 0.1-0.3 (focused), 0.7-1.0 (balanced), 1.0-2.0 (creative)
- **top_k**: Number of tokens to consider
- **top_p**: Nucleus sampling threshold
- **repetition_penalty**: >1.0 discourages repetition

## Example
```bash
# Use config prompt
python -m app model generate --config configs/small_train.yaml

# Custom prompt
python -m app model generate --config configs/small_train.yaml --prompt "Science is"
```

## Tips
- Lower temperature for more predictable text
- Use repetition penalty (1.1-1.3) to reduce loops
- Adjust top-k/top-p for diversity vs quality