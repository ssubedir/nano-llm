# WikiText-103 Raw Dataset

WikiText-103 is a collection of over 100 million tokens extracted from Good and Featured articles on Wikipedia. The raw version contains the original, untokenized text.

## Dataset Details

- **Size**: ~103 million words
- **Source**: Wikipedia Good and Featured articles
- **Format**: Raw text files
- **Language**: English

## Files

- `wiki.train.raw` - Training set
- `wiki.valid.raw` - Validation set 
- `wiki.test.raw` - Test set

## Download Instructions

### Option 1: Using wget / curl
```bash
wget -O wikitext-103-raw-v1.zip https://wikitext.smerity.com/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

```bash
curl -L -o wikitext-103-raw-v1.zip https://wikitext.smerity.com/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

### Option 2: Manual Download
1. Visit: https://wikitext.smerity.com/wikitext-103-raw-v1.zip
2. Download and extract the zip file
3. Place the `wikitext-103-raw` folder in this directory

## Usage in Configuration

Reference these files in your nano-llm configuration:

```yaml
train:
  data_files: ["dataset/wikitext-103-raw-v1/wikitext-103-raw/wiki.train.raw"]

eval:
  data_files: ["dataset/wikitext-103-raw-v1/wikitext-103-raw/wiki.valid.raw"]
```

## Notes

- This dataset retains original capitalization, punctuation, and numbers
- Suitable for training language models that need to handle real-world text
- No preprocessing has been applied, allowing you to customize preprocessing as needed