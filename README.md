# seq2seq-from-scratch
# Neural Machine Translation using Attention Mechanism

A Deep Learning-based Neural Machine Translation (NMT) system built using PyTorch. This project implements the Sequence-to-Sequence (Seq2Seq) architecture with Bahdanau Attention to improve translation quality by allowing the decoder to focus on relevant parts of the input sequence during translation.

---

## Project Overview

Traditional Seq2Seq models compress the entire source sentence into a single context vector, which can lead to information loss for longer sentences.

To overcome this limitation, this project incorporates an Attention Mechanism that enables the decoder to dynamically attend to different encoder hidden states while generating each target word.

The model is trained to translate sentences from a source language to a target language and demonstrates how attention improves translation performance.

---

## Features

- Seq2Seq Encoder-Decoder Architecture
- Bahdanau Attention Mechanism
- Teacher Forcing during training
- Custom Dataset and DataLoader
- Tokenization and Vocabulary Building
- Padding and Sequence Handling
- Training and Validation Loops
- Loss Visualization
- Inference for Sentence Translation
- PyTorch Implementation

---

## Model Architecture

### Encoder

The encoder processes the input sequence and generates hidden states for each input token.

Components:
- Embedding Layer
- LSTM/GRU Network
- Hidden States

### Attention Layer

The attention mechanism calculates attention scores between the current decoder hidden state and all encoder hidden states.

Steps:

1. Calculate Alignment Scores
2. Apply Softmax
3. Generate Attention Weights
4. Compute Context Vector

The context vector contains the most relevant information from the source sentence.

### Decoder

The decoder uses:
- Previous predicted token
- Context Vector
- Decoder Hidden State

to generate the next target token.

---

## Attention Mechanism

Instead of relying on a single context vector:

\[
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

The decoder learns which source words are most relevant for predicting the next target word.

Benefits:

- Better long-sequence handling
- Improved translation accuracy
- Better alignment between source and target words

---

## Dataset

The model can be trained on any parallel corpus.

Example:

| Source Language | Target Language |
|----------------|----------------|
| Hello | नमस्ते |
| How are you? | आप कैसे हैं? |
| I love machine learning | मुझे मशीन लर्निंग पसंद है |

Dataset Format:

```csv
source,target
Hello,नमस्ते
How are you?,आप कैसे हैं?
Realistic Seq2Seq + Attention training accuracy.png

