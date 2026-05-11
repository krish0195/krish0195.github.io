---
title: "Transformer 6: Transformer Architecture & Mechanics"
date: 2026-05-12
draft: false
description: "Breaking down the matrix operations behind Query, Key, and Value vectors and the importance of the scaling factor."
tags: ["Transformer", "Mathematics", "Linear Algebra", "QKV"]
showToc: true
---\

# Deep Dive: Transformer Architecture & Mechanics
**Class Date:** May 3, 2026  
**Topic:** Positional Encoding, Self-Attention, and Layer Normalization

---

## 1. The Core Challenge: Parallelism
[cite_start]Transformers process all tokens in a sentence simultaneously (in parallel)[cite: 30]. [cite_start]While this makes them incredibly fast compared to older models, it creates a problem: the model does not inherently know the order of words[cite: 31]. [cite_start]For example, without extra info, "Dog bites man" and "man bites dog" look exactly the same to the model[cite: 36, 37].

### The Solution: Positional Encoding (PE)
[cite_start]Positional Encoding is a technique used to inject information about the order of tokens into their embeddings[cite: 28].

* [cite_start]**Mechanism:** We add a PE vector to the original word embedding[cite: 81, 83].
* [cite_start]**Dimension:** The PE vector must have the same dimension as the embedding ($d_{model}$)[cite: 85, 146].
* [cite_start]**Formula:** We use Sine and Cosine functions of different frequencies[cite: 132]:
  - $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
  - $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$

---

## 2. Self-Attention Mechanism
[cite_start]Self-attention allows each word in a sentence to "attend" to every other word to find relevant context[cite: 388].

### The QKV Transformation
[cite_start]To compute attention, every input embedding is transformed into three distinct vectors using learnable weight matrices ($W_Q, W_K, W_V$)[cite: 521, 522, 673, 674]:
1. [cite_start]**Query (Q):** What I am looking for[cite: 703].
2. [cite_start]**Key (K):** What I contain/offer[cite: 705].
3. [cite_start]**Value (V):** The actual information I provide[cite: 706].

### [cite_start]Step-by-Step Calculation[cite: 957]:
1. [cite_start]**Compute Scores:** Perform a Dot Product between Query and Key ($Q \cdot K^T$)[cite: 222, 998].
2. [cite_start]**Scaling:** Divide the score by $\sqrt{d_k}$ (the dimension of the key vector) to ensure training stability and smooth gradients[cite: 680, 692, 697].
3. [cite_start]**Softmax:** Apply a Softmax layer to convert scores into probabilities (weights) that sum to 1[cite: 700, 715].
4. [cite_start]**Weighted Sum:** Multiply these weights by the Value vector (V) to get the final **Contextual Vector**[cite: 721, 722].



---

## 3. Multi-Head Attention
[cite_start]A single "head" of attention can only capture one perspective at a time[cite: 738]. [cite_start]Multi-Head Attention uses multiple sets of $Q, K, V$ matrices in parallel to capture various relationships simultaneously[cite: 726, 732].

* [cite_start]**Example of Ambiguity:** "The chicken is ready to eat"[cite: 741]. 
    - [cite_start]Head 1 might focus on the chicken being **cooked**[cite: 743].
    - [cite_start]Head 2 might focus on the chicken being **hungry**[cite: 744].
* [cite_start]**Original Research Paper (ORP):** Typically uses **8 heads**[cite: 750, 809].

---

## 4. Layer Normalization
[cite_start]Layer Normalization standardizes the features of each vector to improve training stability[cite: 1067]. 

[cite_start]**Formula:** $\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ [cite: 1070]

* [cite_start]**$\mu$ (Mean):** Centers the data[cite: 1074, 1092].
* [cite_start]**$\sigma^2$ (Variance):** Scales the data[cite: 1075, 1097].
* [cite_start]**$\gamma$ & $\beta$:** Learnable parameters that allow the model to scale and shift the normalized values as needed[cite: 1077, 1078].



---

## 5. Cross-Attention
[cite_start]While self-attention looks within the same sequence, **Cross-Attention** allows the Decoder to gather information from the Encoder[cite: 897]. 
* [cite_start]**Queries (Q):** Come from the Decoder (output sequence)[cite: 898].
* [cite_start]**Keys (K) & Values (V):** Come from the Encoder (input sequence)[cite: 898].

---
*Summary of end-to-end data flow in Transformers based on Class-13 notes.*