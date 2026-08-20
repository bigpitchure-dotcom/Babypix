# BabyPix1

A tiny language model built from scratch for educational purposes.

**Version:** 0.1  
**Architecture:** GPT-style decoder-only Transformer  
**Parameters:** ~13M (configurable)

---

## What is BabyPix1?

BabyPix1 is a minimal implementation of a modern language model. Unlike using pretrained models like GPT-2 or LLaMA, BabyPix1 trains **from scratch** with randomly initialized weights, so you can see exactly how every component works.

### Goals

- Learn how Transformers work by building one from the ground up
- Keep the model small enough to train on consumer hardware
- Provide clear, beginner-friendly code
- Build incrementally — test each stage before moving forward

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Tokenizer

```bash
# Train on sample data
python scripts/build_vocab.py --data data/sample.txt --vocab_size 2000

# Train on your own data
python scripts/build_vocab.py --data data/your_corpus.txt --vocab_size 5000
```

### 3. Encode Text

```bash
# Encode a string
python scripts/encode.py --text "Hello world"

# Show individual tokens
python scripts/encode.py --text "Hello world" --show_tokens
```

### 4. Decode Token IDs

```bash
# Decode space-separated IDs
python scripts/decode.py --ids "5 12 34 67"

# Decode list format
python scripts/decode.py --ids "[5, 12, 34, 67]"
```

### 5. Run Tokenizer Tests

```bash
python -m pytest tests/test_tokenizer.py -v
```

### 6. Train the Model (Coming in v0.2)

```bash
python scripts/train.py --config config/default.yaml
```

### 7. Generate Text (Coming in v0.2)

```bash
python scripts/generate.py --checkpoint checkpoints/epoch_10.pt --prompt "Hello"
```

---

## Project Structure

```
BabyPix1/
├── babypix1/               # Core library
│   ├── __init__.py         # Package init
│   ├── config.py           # Configuration
│   ├── utils.py            # Helper functions
│   ├── tokenizer.py        # BPE tokenizer
│   ├── model.py            # Transformer architecture (coming soon)
│   ├── dataset.py          # Data loading (coming soon)
│   ├── trainer.py          # Training loop (coming soon)
│   ├── generator.py        # Text generation (coming soon)
│   └── checkpoint.py       # Model persistence (coming soon)
├── scripts/                # Entry points
│   ├── build_vocab.py      # Tokenizer training
│   ├── encode.py           # Encode text
│   ├── decode.py           # Decode token IDs
│   ├── train.py            # Training script (coming soon)
│   └── generate.py         # Inference script (coming soon)
├── config/
│   └── default.yaml        # Default settings
├── data/
│   └── sample.txt          # Sample training data
├── checkpoints/            # Saved models
├── tests/
│   └── test_tokenizer.py   # Tokenizer tests
├── ARCHITECTURE.md         # Technical architecture
├── PRD.md                  # Product requirements
└── README.md               # This file
```

---

## Tokenizer

The tokenizer converts raw text into integer token IDs that the model can process.

### Features

- **BPE (Byte-Pair Encoding):** Learns subword units from your data
- **Configurable vocab size:** Default 2000, adjust as needed
- **Special tokens:**
  - `<pad>` — Padding token
  - `<unk>` — Unknown token
  - `<bos>` — Beginning of sequence
  - `<eos>` — End of sequence

### How It Works

```
"Hello world"
    ↓
Tokenizer (BPE)
    ↓
[<bos>, 152, 432, <eos>]
```

### Commands

```bash
# Train tokenizer on text files
python scripts/build_vocab.py --data data/sample.txt --vocab_size 2000

# Encode text to token IDs
python scripts/encode.py --text "Hello world" --show_tokens

# Decode token IDs back to text
python scripts/decode.py --ids "152 432"
```

---

## Model Architecture (Coming Soon)

```
Input Text
    ↓
Tokenizer (BPE)
    ↓
Token IDs
    ↓
Token Embeddings + Positional Embeddings
    ↓
Transformer Block × 6
    ↓
Language Model Head
    ↓
Next Token Probabilities
```

### Default Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 2,000 |
| Embedding dimension | 256 |
| Attention heads | 8 |
| Feed-forward dimension | 1,024 |
| Transformer layers | 6 |
| Max sequence length | 512 |
| Total parameters | ~13.4M |

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Hardware

| Setup | Training Time |
|-------|---------------|
| CPU (8 cores) | ~24 hours |
| GPU (GTX 1060) | ~2 hours |
| GPU (RTX 3080) | ~30 minutes |

---

## Learning Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual explanation
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT architecture

---

## License

MIT

---

## Contributing

This is an educational project. Feel free to experiment, break things, and learn!
