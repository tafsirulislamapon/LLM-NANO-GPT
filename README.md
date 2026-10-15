# LLM from Scratch

Building a large language model from the ground up in Python and PyTorch, one step at a time. It starts with the simplest possible language model, a **character-level bigram model**, and works up toward a GPT-style Transformer.

The repo is public so anyone can follow along, learn from it, or suggest improvements.

> **Status:** Early stage. The data is in place and the first notebook loads the training text. The bigram model is being built now.

---

## Project structure

```
LLM/
├── bigram.ipynb        # Notebook: character-level bigram language model (in progress)
├── wizard_of_oz.txt    # Training text: "Dorothy and the Wizard in Oz" by L. Frank Baum (~238 KB)
├── dataset.txt         # Larger training text: Tiny Shakespeare (~1.1 MB, 40,000 lines)
└── README.md
```

`cuda/` is a local Python virtual environment and is not committed.

## Datasets

| File | Content | Size | Source |
|------|---------|------|--------|
| `wizard_of_oz.txt` | *Dorothy and the Wizard in Oz* (1908), L. Frank Baum | ~238 KB | Project Gutenberg (public domain) |
| `dataset.txt` | Tiny Shakespeare: lines from Shakespeare's plays | ~1.1 MB | [char-rnn](https://github.com/karpathy/char-rnn) |

Both are plain UTF-8 text. The model reads them one character at a time, so no external tokenizer is needed.

## How it works

### 1. Bigram language model (current step)

A bigram model predicts the next character using only the current one. It is the simplest language model and provides the base for the full pipeline:

1. **Load the text** and build the vocabulary: every unique character in the dataset.
2. **Tokenize** with two lookup tables, `stoi` (character → integer) and `itos` (integer → character), to encode and decode text.
3. **Split the data** into training and validation sets (for example 80/20).
4. **Batch** the data into random chunks of `block_size` characters, where the target is the input shifted one position to the right.
5. **Model:** an `nn.Embedding(vocab_size, vocab_size)` table whose row for each character holds the logits for the next character.
6. **Train** with cross-entropy loss and the AdamW optimizer, on the GPU when one is available.
7. **Generate** new text by repeatedly sampling the next character from the predicted distribution.

### 2. Roadmap

- [x] Load and inspect the training text
- [ ] Character-level tokenizer (encode / decode)
- [ ] Train/validation split and batching
- [ ] Bigram model: training loop, loss estimation, text generation
- [ ] Self-attention (single head)
- [ ] Multi-head attention
- [ ] Feed-forward layers, residual connections, LayerNorm
- [ ] Full GPT-style Transformer (decoder-only)
- [ ] Train on a larger corpus
- [ ] Save/load checkpoints and an interactive chat/prompt script

## Getting started

### Requirements

- Python 3.10+ (developed on Python 3.14)
- [PyTorch](https://pytorch.org/get-started/locally/) (CUDA build recommended; developed with `torch 2.14 + cu132`)
- NumPy, Matplotlib
- Jupyter

An NVIDIA GPU with CUDA speeds up training a lot, but the code also runs on CPU.

### Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd LLM

# Create and activate a virtual environment
python -m venv cuda
# Windows
cuda\Scripts\activate
# macOS / Linux
source cuda/bin/activate

# Install PyTorch (pick the command for your system at https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# Install the rest
pip install numpy matplotlib jupyter ipykernel

# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name cuda --display-name "cuda"
```

### Run

```bash
jupyter notebook bigram.ipynb
```

Choose the `cuda` kernel and run the cells from top to bottom.

To check that PyTorch can see your GPU:

```python
import torch
print(torch.cuda.is_available())  # True if CUDA is set up correctly
```

## Why build an LLM from scratch?

Calling an API is easy. Understanding what happens inside the model is harder. Building each piece by hand (tokenizer, embeddings, attention, training loop) shows how models like GPT actually work, and why they behave the way they do.

## Acknowledgements

- Andrej Karpathy's [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) and the [nanoGPT](https://github.com/karpathy/nanoGPT) project
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- [Project Gutenberg](https://www.gutenberg.org/) for the public-domain text

## Contributing

This is a personal learning project, but issues, suggestions and pull requests are welcome. If you spot a bug or know a better way to explain something, open an issue.

## License

No license has been chosen yet. Add one (MIT is a common choice for open learning projects) to let others reuse the code.
