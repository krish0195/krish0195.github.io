---
title: "Transformer 3: Deep Dive into Encoder Components"
date: 2026-05-10
draft: false
description: "Exploring the power of stacking layers, the mechanics of residual connections, and why the FFN is critical for non-linear pattern recognition."
tags: ["Transformer", "Deep Learning", "Encoder", "AI Architecture"]
showToc: true
---

## 1. The Power of Stacking (Why 6 Layers?)
In the original Transformer research paper, the authors used **6 encoder layers** and **6 decoder layers**. However, this number is a **Hyperparameter** and is not fixed; it is determined experimentally based on the task.

* **Vertical Stacking:** Stacking layers vertically allows the model to capture deeper relationships and build hierarchical meaning.
* **Hierarchical Understanding:**
    * **Layer 1:** Learns basic word relations.
    * **Layer 3:** Begins to understand phrase meanings.
    * **Layer 6:** Achieves full sentence-level understanding.
    * **Deep Context:** Modern models like BERT and GPT stack 12 to 96+ layers to achieve deep contextual reasoning.

---

## 2. Residual Connections (The Skip Connection)
As neural networks grow deeper, they often suffer from **Vanishing Gradients**, where weights become too small, training becomes unstable, and information is lost.

[Image of residual connection in a transformer block]

* **The Solution:** Residual connections (also known as skip connections) allow the model to learn a "small improvement over the input" rather than a completely new transformation.
* **Impact:** * It ensures a better gradient flow during backpropagation.
    * It prevents information loss, allowing the model to retain "identical info" as it passes through the stack.
    * It enables the stable training of massive stacks like GPT-3 or DeepSeek.

---

## 3. Feed-Forward Neural Networks (FFN)
While Self-Attention is excellent at learning relationships *across* different tokens, the FFN operates on each token **independently**.

* **The Architecture:** In the original research, the FFN consists of two linear transformations with a ReLU activation in between.
* **Dimensions:** It typically expands the vector from **512 to 2048** dimensions and back to **512**.
* **Key Role:** The FFN adds