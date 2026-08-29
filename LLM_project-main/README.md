# my-first-llm

> A from-scratch GPT-style language model you can train, chat with, evaluate, and deploy — all from your own machine.

Created with [create-llm](https://github.com/theaniketgiri/create-llm) ✨

---

## What Is This?

This is a complete, self-contained framework for training a GPT-style large language model on your own text data. It covers the entire lifecycle — tokenizer training, data preprocessing, model training, evaluation, interactive chat, and deployment — with every component written in Python and PyTorch.

The default configuration targets a **nano model (~500K parameters)** that runs on any CPU and trains in under 2 minutes. You can scale up to a ~1B parameter base model when you have the hardware for it.

---

## Project Overview

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only GPT Transformer |
| **Default Template** | NANO (~500K parameters) |
| **Tokenizer** | BPE (Byte-Pair Encoding) |
| **Training Framework** | PyTorch |
| **Config Format** | `llm.config.js` (Node.js module) |
| **Default Hardware** | CPU-friendly (no GPU required) |
| **Default Training Time** | 1–2 minutes |
| **Min Data Needed** | 100+ examples |

---

## Architecture

The model is a **decoder-only transformer** (GPT-style) built from scratch using PyTorch. Key design choices:

- **Pre-LayerNorm residuals** — LayerNorm is applied before each sub-layer (attention, FFN), not after. This improves training stability.
- **Multi-head causal self-attention** — A fused QKV projection with a registered causal mask buffer prevents the model from attending to future tokens.
- **Position-wise FFN** — Two-layer MLP with GELU activation and 4× hidden dimension expansion.
- **Learned positional embeddings** — Absolute position embeddings added to token embeddings at the input.
- **Weight tying** — The input token embedding matrix is shared with the output projection head, reducing parameter count and typically improving language modeling.
- **Autoregressive generation** — Supports temperature scaling, top-k filtering, and top-p (nucleus) sampling.

### Model Size Variants

Four pre-configured sizes are available, selectable via `model.size` in `llm.config.js`:

| Size   | Layers | Heads | Dim   | Max Seq Len | Parameters | Hardware                  |
|--------|--------|-------|-------|-------------|------------|---------------------------|
| nano   | 3      | 4     | 128   | 256         | ~500K      | Any CPU, 2 GB RAM         |
| tiny   | 4      | 4     | 256   | 512         | ~5M        | CPU or basic GPU          |
| small  | 12     | 12    | 768   | 1024        | ~100M      | RTX 3060+ or equivalent   |
| base   | 24     | 16    | 1536  | 2048        | ~1B        | A100 / 2× RTX 4090        |

---

## Project Structure

```
my-first-llm/
│
├── 📂 data/
│   ├── raw/                    # ← Put your .txt files here
│   ├── processed/              # Tokenized tensors (auto-generated)
│   ├── dataset.py              # LLMDataset + StreamingLLMDataset (PyTorch)
│   └── prepare.py              # Sliding-window tokenization + train/val split
│
├── 📂 models/
│   ├── architectures/
│   │   ├── gpt.py             # Core GPT model (attention, FFN, blocks, generate)
│   │   ├── nano.py            # ~500K parameter config
│   │   ├── tiny.py            # ~5M parameter config
│   │   ├── small.py           # ~100M parameter config
│   │   └── base.py            # ~1B parameter config
│   ├── config.py              # ConfigLoader: validates llm.config.js, auto-syncs vocab
│   └── __init__.py
│
├── 📂 tokenizer/
│   ├── train.py               # BPE / WordPiece / Unigram tokenizer training
│   └── tokenizer.json         # Trained tokenizer (auto-generated)
│
├── 📂 training/
│   ├── train.py               # Main entry point ⭐
│   ├── trainer.py             # Training loop: AMP, grad accum, LR schedule, callbacks
│   └── callbacks/
│       ├── checkpoint.py      # Rolling checkpoint saves + best-model tracking
│       ├── logging.py         # CSV + TensorBoard logging
│       └── checkpoint_manager.py
│
├── 📂 evaluation/
│   ├── evaluate.py            # Perplexity + loss evaluation on validation set
│   └── generate.py            # Text generation with top-k / top-p sampling
│
├── 📂 plugins/                # Optional integrations
│   ├── plugin_manager.py      # Dynamic plugin loader
│   ├── wandb_plugin.py        # Weights & Biases experiment tracking
│   ├── huggingface_plugin.py  # Hugging Face Hub upload
│   └── example_plugin.py     # Template for writing your own plugin
│
├── 📂 utils/
│   ├── exceptions.py          # Typed exception hierarchy with suggestions
│   └── handlers.py            # Retry decorators, safe checkpoint save, NaN loss guard
│
├── 📂 tests/                  # Test suite
│
├── 📄 llm.config.js           # Central configuration for everything ⚙️
├── 📄 requirements.txt        # Python dependencies
├── 📄 chat.py                 # Terminal chat (with /temp, /clear, /quit commands)
├── 📄 chat_interface.py       # Gradio web chat UI at localhost:7860
├── 📄 deploy.py               # Deploy to Hugging Face Hub or Replicate
└── 📄 compare.py              # Side-by-side comparison of multiple checkpoints
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Node.js is also required to parse `llm.config.js`. Download it from [nodejs.org](https://nodejs.org/) if you don't have it.

### 2. Add training data

Place plain text `.txt` files in `data/raw/`:

```bash
# Example: download Project Gutenberg Shakespeare
curl https://www.gutenberg.org/files/100/100-0.txt -o data/raw/shakespeare.txt

# Or use your own files
cp /path/to/your/data.txt data/raw/
```

Data requirements: plain text, UTF-8 encoding, at least 100 lines.

### 3. Train the tokenizer

```bash
python tokenizer/train.py --data data/raw/
```

This trains a BPE tokenizer on your data and saves it to `tokenizer/tokenizer.json`. The vocabulary size is automatically picked up by the model — no manual syncing needed.

### 4. Prepare the dataset

```bash
python data/prepare.py
```

Tokenizes your text using a sliding window, splits it into train/val sets, and saves them as PyTorch tensors in `data/processed/`.

### 5. Train the model

```bash
python training/train.py
```

The script validates the config, checks that tokenizer and model vocabulary sizes match, and starts training. You'll see real-time loss, learning rate, tokens/sec, and estimated time remaining.

After training completes, a menu lets you:
1. Continue training for more steps
2. Launch the web chat interface
3. Exit

### 6. Chat with your model

```bash
# Web UI (Gradio) — opens at http://localhost:7860
python chat_interface.py

# Terminal chat
python chat.py --checkpoint checkpoints/checkpoint-best.pt
```

Terminal chat commands: `/temp <value>` to change temperature, `/clear` to reset context, `/quit` to exit.

---

## Configuration

All settings live in `llm.config.js`. Edit this file to change anything about how the model is built or trained.

```javascript
module.exports = {
  model: {
    type: 'gpt',          // Architecture (only gpt supported currently)
    size: 'nano',         // nano | tiny | small | base
    vocab_size: 5000,     // Auto-synced from tokenizer — safe to leave as-is
    max_length: 256,      // Maximum sequence length
    layers: 3,
    heads: 4,
    dim: 128,
    dropout: 0.1,
  },
  training: {
    batch_size: 8,
    learning_rate: 0.0005,
    warmup_steps: 100,    // Linear warmup before cosine decay
    max_steps: 1000,
    eval_interval: 200,
    save_interval: 500,
    optimizer: 'adamw',   // adamw | adam | sgd
    weight_decay: 0.01,
    gradient_clip: 1.0,
    mixed_precision: false,
    gradient_accumulation_steps: 1,
  },
  data: {
    max_length: 256,
    stride: 128,          // Sliding window overlap
    val_split: 0.1,
    shuffle: true,
  },
  tokenizer: {
    type: 'bpe',          // bpe | wordpiece | unigram
    vocab_size: 5000,
    min_frequency: 2,
    special_tokens: ['<pad>', '<unk>', '<s>', '</s>'],
  },
};
```

**Common adjustments:**

| Goal | What to change |
|------|----------------|
| Less memory usage | Reduce `batch_size`; enable `mixed_precision` |
| Simulate larger batch | Increase `gradient_accumulation_steps` |
| Better quality | Increase `max_steps`; add more training data |
| Reduce overfitting | Increase `dropout` to 0.2–0.4; add more data |
| Unstable loss | Lower `learning_rate`; increase `warmup_steps` |

---

## Training Features

The `Trainer` class (`training/trainer.py`) includes:

- **Cosine LR schedule with linear warmup** — Learning rate ramps up linearly for `warmup_steps`, then decays via cosine schedule.
- **Automatic mixed precision (AMP)** — FP16 training with `torch.cuda.amp` when `mixed_precision: true`.
- **Gradient accumulation** — Simulates larger batch sizes without extra memory.
- **Gradient clipping** — Prevents exploding gradients (default threshold: 1.0).
- **Callback system** — `CheckpointCallback` and `LoggingCallback` are wired in automatically. You can add custom callbacks.
- **Rolling checkpoint management** — Keeps the `N` most recent checkpoints (configurable via `save_total_limit`) and always keeps `checkpoint-best.pt` (lowest validation loss).
- **TensorBoard logging** — Loss, learning rate, and perplexity are logged to `logs/` and viewable with `tensorboard --logdir logs`.

---

## Evaluation & Generation

```bash
# Evaluate a checkpoint — reports loss and perplexity
python evaluation/evaluate.py --checkpoint checkpoints/checkpoint-best.pt

# Generate text from a prompt
python evaluation/generate.py \
  --checkpoint checkpoints/checkpoint-best.pt \
  --prompt "Once upon a time" \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95 \
  --max-length 200
```

**Temperature guide:**
- `0.1–0.3` — Focused and deterministic output
- `0.7–0.9` — Balanced creativity
- `1.0–1.5` — High diversity, more random

---

## Advanced Usage

### Resume from checkpoint

```bash
python training/train.py --resume checkpoints/checkpoint-500.pt
```

### Override training steps

```bash
python training/train.py --max-steps 5000
```

### Use a custom config file

```bash
python training/train.py --config my-experiment.js
```

### Compare multiple checkpoints

```bash
python compare.py checkpoints/checkpoint-500.pt checkpoints/checkpoint-1000.pt
```

Generates a side-by-side table of loss, perplexity, and sample output for each checkpoint, and saves a JSON report to `logs/`.

### Deploy to Hugging Face Hub

```bash
python deploy.py \
  --checkpoint checkpoints/checkpoint-best.pt \
  --to huggingface \
  --repo-id username/my-model
```

---

## Plugins

Optional integrations can be enabled in `llm.config.js`:

```javascript
plugins: [
  'wandb',        // Weights & Biases experiment tracking
  'huggingface',  // Auto-upload checkpoints to Hugging Face Hub
]
```

The plugin manager loads them at startup and calls lifecycle hooks during training (`on_step_end`, `on_checkpoint_save`, etc.). Plugins that fail to initialize are skipped with a warning — training continues regardless.

---

## Hardware Requirements

| Template | RAM  | GPU          | Training Time |
|----------|------|--------------|---------------|
| nano     | 2 GB | None needed  | 1–2 minutes   |
| tiny     | 4 GB | Optional     | 10–30 minutes |
| small    | 16 GB | RTX 3060+   | Hours         |
| base     | 80 GB+ | A100 / 4090s | Days        |

---

## Troubleshooting

**"Vocab size mismatch detected"**
The system auto-corrects this. It always reads the actual vocabulary size from `tokenizer/tokenizer.json` and uses that for the model — no manual fix needed.

**Repetitive output ("which which which...")**
This usually means the tokenizer and model were mismatched. Retrain the tokenizer, then retrain the model from scratch.

**"CUDA out of memory"**
Reduce `batch_size`, enable `mixed_precision: true`, or increase `gradient_accumulation_steps`.

**"Tokenizer not found"**
Run `python tokenizer/train.py --data data/raw/` before training.

**"Training data not found"**
Run `python data/prepare.py` before training.

**Loss not decreasing**
Try adjusting `learning_rate` (range: `1e-4` to `1e-3`) or increasing `warmup_steps`.

**Perplexity < 1.5 / Overfitting warning**
Add more training data, use a smaller model size, or increase `dropout`.

---

## Dependencies

Core dependencies (see `requirements.txt` for pinned versions):

- **PyTorch** — Model and training
- **tokenizers** — HuggingFace fast tokenizer library (BPE/WordPiece/Unigram)
- **Gradio** — Web chat interface
- **TensorBoard** — Training visualization
- **tqdm** — Progress bars
- **tabulate** — Comparison output formatting

Optional (enabled via plugins):
- **wandb** — Experiment tracking
- **huggingface_hub** — Model/tokenizer upload

---

## Resources

- [create-llm on GitHub](https://github.com/theaniketgiri/create-llm)
- [PyTorch documentation](https://pytorch.org/docs/)
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/)
- [TensorBoard guide](https://www.tensorflow.org/tensorboard/get_started)

---

## License

MIT

---

Built with [create-llm](https://github.com/theaniketgiri/create-llm) — PyTorch · Tokenizers · Gradio · TensorBoard
