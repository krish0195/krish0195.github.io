---
title: "Transformer 5: The QKV Mathematical Engine"
date: 2026-05-12
draft: false
description: "Breaking down the matrix operations behind Query, Key, and Value vectors and the importance of the scaling factor."
tags: ["Transformer", "Mathematics", "Linear Algebra", "QKV"]
showToc: true
---

## Learnable Parameters
The magic of the Transformer lies in the three weight matrices: $W_Q, W_K,$ and $W_V$. These are not fixed; they are **learnable parameters** that the model refines over thousands of training epochs to better understand context.

## The Attention Flow
1. **Linear Transformation:** Project the 512-dim embedding into 64-dim Q, K, and V vectors.
2. **Scoring:** Dot product of Query and Key to find similarity.
3. **Scaling:** Divide by $\sqrt{d_k}$ to prevent gradient vanishing.
4. **Softmax:** Convert scores into weights that sum to 1.
5. **Contextual Output:** Multiply weights by the Value vectors to produce the final contextual representation.

This process allows the word "bank" to physically move its position in vector space closer to "money" or "river" depending on the sentence.