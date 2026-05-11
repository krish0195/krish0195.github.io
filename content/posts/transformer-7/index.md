
---
title: "Transformer 7: he Transformer Decoder: Mechanics and Autoregressive Inference"
date: 2026-05-12
draft: false
description: "**Topic:** Decoder Architecture, Masked Self-Attention, and Cross-Attention."
tags: ["Transformer", "Mathematics", "Linear Algebra", "QKV"]
showToc: true
---\

# The Transformer Decoder: Mechanics and Autoregressive Inference
**Class Date:** May 9, 2026  
**Topic:** Decoder Architecture, Masked Self-Attention, and Cross-Attention

---

## 1. Overview of the Decoder
While the Encoder reads and understands the source sentence, the Decoder is responsible for generating the translated target sentence word by word. In the original research paper, the Decoder consists of a stack of **6 identical layers** ($N_x=6$).

### The Decoder Block Components:
Each layer in the Decoder contains three primary sub-layers:
1. **Masked Multi-Head Self-Attention:** Blocks future tokens so the model only sees the past.
2. **Multi-Head Cross-Attention:** Gathers relevant information from the Encoder's output.
3. **Feed-Forward Neural Network (FFNN):** Processes the combined information for the next layer.

[Image of Transformer Decoder architecture highlighting Masked Self-Attention, Cross-Attention, and FFNN]

---

## 2. Masked Self-Attention
Masked self-attention is a specific type of attention where future tokens are blocked so that each token can only attend to itself and previous tokens.

* **Why it is needed:** During training, we feed the entire target sentence in parallel. Without masking, the model would "see" the answer it is supposed to predict (Data Leakage).
* **The Mechanism:** Future tokens are set to $-\infty$ before the Softmax layer. This ensures the Softmax output for those positions is **0**, effectively hiding them.

---

## 3. Cross-Attention: Bridging the Gap
Cross-attention is the mechanism where the Decoder attends to the Encoder's output sequence to gather relevant information.

* **Queries (Q):** Come from the Decoder's previous masked attention layer (the output sequence).
* **Keys (K) & Values (V):** Come from the Encoder (the input sequence).
* **Purpose:** It allows the model to focus on specific parts of the input sentence while generating each word of the output.

[Image of cross-attention mechanism showing Queries from Decoder and Keys/Values from Encoder]

---

## 4. Training vs. Inference
There is a fundamental difference in how the Decoder operates depending on the stage:

| Feature | Training Stage | Inference (Prediction) Stage |
| :--- | :--- | :--- |