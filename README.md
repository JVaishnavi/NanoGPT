# NanoGPT: Building GPT from Scratch

This repository contains a PyTorch implementation of a **Decoder-only Transformer** model, built from scratch.

The model is trained on the **Tiny Shakespeare** dataset to generate infinite Shakespeare-like text at the character level.

## Project Overview

The goal of this project was to demystify Large Language Models (LLMs) by building one piece by piece. Starting from a simple Bigram model, the architecture was iteratively upgraded to a full-scale GPT model containing:
* **Self-Attention & Multi-Head Attention**
* **Feed-Forward Networks**
* **Residual Connections**
* **Layer Normalization**

## Model Architecture

Unlike the original Transformer (which uses an Encoder-Decoder architecture for translation), this model uses a **Decoder-only** architecture, which is the standard for generative text models like GPT-3 and ChatGPT.



### Key Components Implemented
1.  **Token & Positional Embeddings:** Encodes characters into learnable vectors and injects positional information.
2.  **Causal Self-Attention:** Ensures the model can only attend to *past* tokens (using a triangular mask), preserving the auto-regressive property required for text generation.
3.  **Multi-Head Attention:** Runs multiple attention heads in parallel to capture different types of relationships in the text.
4.  **Feed-Forward Blocks:** Adds computation capacity (MLP) after the communication (Attention) phase.
5.  **Residual Connections & LayerNorm:** Facilitates deep network training by allowing gradients to flow unimpeded.

## Dataset: Tiny Shakespeare

* **Source:** A concatenation of Shakespeare's works.
* **Size:** ~1MB (approx. 1 million characters).
* **Vocabulary:** 65 unique characters (including punctuation and newlines).
* **Tokenizer:** Character-level tokenizer (maps characters to integers).

## Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/nanogpt-shakespeare.git](https://github.com/yourusername/nanogpt-shakespeare.git)
cd nanogpt-shakespeare
