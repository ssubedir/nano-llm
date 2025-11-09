# Datasets

This folder contains datasets used for training and evaluating the nano-llm models.

## Available Datasets

### WikiText-103 Raw
- Location: [`wikitext-103-raw-v1/`](./wikitext-103-raw-v1/)
- Description: A large-scale language modeling dataset based on Wikipedia articles
- Size: ~103 million words
- Files: Training, validation, and test sets

See individual dataset folders for specific download and usage instructions.

To use these datasets with nano-llm, reference them in your configuration file:

```yaml
train:
  data_files: ["dataset/wikitext-103-raw-v1/wikitext-103-raw/wiki.train.raw"]
```

For more information about data processing and configuration, see the main documentation.