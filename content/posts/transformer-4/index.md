---
title: "Transformer 4: Positional Encoding & The Math of Self-Attention"
date: 2026-05-11
draft: false
description: "Why Transformers need trigonometry to understand word order and a deep dive into the Query, Key, and Value (QKV) mechanism."
tags: ["Transformer", "Positional Encoding", "Self-Attention", "Math"]
showToc: true
---

## Why Positional Encoding?
Unlike RNNs, Transformers process all words simultaneously. This parallelism is great for speed but loses the "sequence" info. Positional Encoding (PE) restores this by adding a unique mathematical signature to each word based on its index.

## The Sine-Cosine Formula
The model uses alternating sine and cosine waves of different frequencies.
* **Even indices:** use Sine.
* **Odd indices:** use Cosine.

This ensures that the distance between any two positions is consistent and interpretable by the Neural Network.

## The Self-Attention Equation
The core of the Transformer is calculated as:
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This formula allows the model to dynamically focus on relevant parts of the input sequence, creating a **Contextual Vector** for every token.