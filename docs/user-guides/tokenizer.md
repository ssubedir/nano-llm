# Tokenizer Command

## Syntax
```bash
python -m app tokenizer --config CONFIG_FILE
```

## Purpose
Trains a Byte-Pair Encoding (BPE) tokenizer on your text data.

## Configuration
```yaml
tokenizer:
  input_files: ["file1.txt", "file2.txt"]  # or data_dir
  output_dir: "tokenizer_output"
  vocab_size: 10000
```

## Example
```bash
python -m app tokenizer --config configs/small_train.yaml
```

This creates tokenizer files in the output directory including `tokenizer.json` and `vocab.json`.

## Tips
- Use 3K-5K vocab for small datasets, 20K+ for large ones
- Include diverse text in your training data
- The same tokenizer must be used for training, evaluation, and generation