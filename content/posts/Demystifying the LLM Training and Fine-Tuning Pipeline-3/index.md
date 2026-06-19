---
title: "Demystifying the LLM Training and Fine-Tuning Pipeline-part-3"
date: 2026-06-06
---
# Fine-Tuning Large Language Models: A Complete Guide

> *From pretraining pipelines to LoRA, QLoRA, and preference alignment — everything you need to understand modern LLM fine-tuning.*

---

## What is Fine-Tuning?

Fine-tuning is the process of taking an **already trained model** and training it further on domain-specific or task-specific data so that it performs better for a particular use case.

Rather than building a model from scratch (which requires billions of dollars, massive compute, and months of training), fine-tuning lets you adapt a powerful open-source model to your own needs in hours or days on a single GPU.

---

## The LLM Training Pipeline

Every major LLM you interact with today — whether from Meta, Mistral, OpenAI, or Deepseek — goes through a structured training pipeline with three stages:

```
Stage 1: Pretraining  →  Stage 2: Instruction Fine-tuning (SFT)  →  Stage 3: Preference Tuning
```

### 1. Pretraining (Unsupervised / Self-supervised Learning)

The model is trained on **raw, internet-scale data** — books, websites, code, articles — without any human labels. The objective is next-token prediction: given a sequence of words, predict the next one.

- **Output**: A "raw" base model (e.g., `meta-llama/Meta-Llama-3-8B`)
- **Data format**: Unstructured text (no instruction/answer pairs)
- **Goal**: Learn general knowledge, grammar, reasoning, world facts

This is the most compute-intensive stage. Training Meta-Llama-3-8B, for example, required **trillions of tokens** and thousands of GPUs.

### 2. Supervised Fine-Tuning — SFT / Instruction Tuning

After pretraining, the raw model knows *a lot*, but it doesn't know how to *follow instructions*. SFT teaches it to respond helpfully given a user prompt.

- **Data format**: Input-Output pairs (IP → OP), e.g., `{"instruction": "Summarize this text", "response": "..."}`
- **Output**: An instruction-tuned model (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`)
- **Goal**: Teach the model conversational behavior, Q&A, chat

This is **supervised learning** — every training example has a correct label (the expected output).

### 3. Preference Tuning (Preference Alignment)

Even after SFT, models can produce outputs that are factually correct but poorly phrased, unsafe, or not aligned with human preferences. Preference tuning fixes this.

The key idea: **collect human feedback** on model outputs (e.g., "Which response do you prefer?"), then retrain the model to produce preferred responses.

Popular preference alignment algorithms:
| Algorithm | Notes |
|-----------|-------|
| **RLHF** (Reinforcement Learning from Human Feedback) | Used by OpenAI; involves a reward model + PPO (Proximal Policy Optimization) |
| **GRPO** | Group Relative Policy Optimization; efficient alternative to PPO |
| **DPO** (Direct Preference Optimization) | Simpler than RLHF; directly optimizes on preference pairs without a reward model |
| **ORPO** | Odds Ratio Preference Optimization; even more efficient variant |

---

## Real-World Example: Pharma Use Case

Imagine you work at a pharmaceutical company and have a large collection of internal documents on molecule studies.

**The problem**: Training an LLM from scratch is extremely difficult and expensive.

**The solution**: Use an open-source model and fine-tune it.

- If your goal is **data generation** (not a chatbot), use the raw base model:
  `meta-llama/Meta-Llama-3-8B` → fine-tune on your pharma dataset → specialized text generator

- If your goal is **Conversational AI / Q&A / Chatbot**, use the instruction-tuned model:
  `meta-llama/Meta-Llama-3-8B-Instruct` → further fine-tune on your IP→OP pairs

---

## Where to Fine-Tune: Hugging Face Ecosystem

[Hugging Face](https://huggingface.co) is the central hub for open-source ML:

- **`datasets` library**: Load thousands of datasets from companies, universities, and the open-source community.
  - **Inbuilt datasets**: Uploaded by researchers — ready to use for common tasks
  - **Custom datasets**: Your own personal or enterprise data

- **Supported data formats**:
  - Standard: `csv`, `xlsx`, `tsv`, `json`
  - Big-data binary formats: `parquet`, `rc`, `orc`

You can also convert your own data to HF-compatible format and upload it publicly so others can use it.

---

## How Transformers Process Data

Before fine-tuning, data goes through a well-defined pipeline inside the Transformer:

```
Raw Text
   ↓
Tokenization (BPE / WordPiece)
   ↓
Token IDs
   ↓
Word Embeddings
   ↓
Positional Encoding
   ↓
Self-Attention
   ↓
FFNN (Feed-Forward Neural Network)
   ↓
Output
```

### Tokenization

Tokens are sub-word units, not whole words. Two dominant algorithms:
- **BPE (Byte Pair Encoding)** — Used by GPT models
- **WordPiece** — Used by BERT

Example:
```
Input:    "I love AI"
Tokens:   ["I", "love", "AI"]
Token IDs: [40, 1842, 9552]
```

### Model Parameters

Every model is defined by its **parameters** — weights and biases in two key components:
- **Self-Attention layers** → Q (Query), K (Key), V (Value) weight matrices
- **FFNN layers** → weight matrices for the feed-forward network

Fine-tuning = updating some or all of these parameters on your new dataset.

---

## BERT Architecture (Encoder-only Models)

BERT is an **encoder-only** model, meaning it processes entire input sequences bidirectionally — unlike GPT which is decoder-only (left-to-right).

**BERT-Base**: 12 Transformer layers, 768 hidden units, 12 attention heads → **110M parameters**
**BERT-Large**: 24 Transformer layers, 1024 hidden units, 16 attention heads → **340M parameters**

Inside BERT's feed-forward network:
- Input dimension: 768 (or 512)
- Hidden dimension: **3072** (4× the input, standard Transformer scaling)
- Output dimension: 768 (or 512)

The `AutoModel` API in Hugging Face loads the raw BERT encoder. To make predictions, you attach a **head** on top (e.g., a classification head for sentiment analysis).

---

## Types of Fine-Tuning

### Parameter Level: Full vs. Partial

#### 1. Full Fine-Tuning
- **Retrain ALL parameters** (weights & biases) of the model
- Requires high GPU power and large infrastructure
- Best performance but most expensive
- Not practical for most use cases

#### 2. Partial Fine-Tuning — PEFT (Parameter-Efficient Fine-Tuning)
- Update only a **subset of parameters**
- Runs on a **single GPU** with small VRAM
- Requires small infrastructure
- Nearly matches full fine-tuning performance for most tasks

PEFT is the go-to approach today for custom fine-tuning.

---

### Old School Partial Fine-Tuning (Pre-LLM era)

Before LLMs, models like CNN (VGG, ResNet), BERT, BART, and T5 were fine-tuned using layer-freezing strategies:

1. **Freeze all layers, retrain only the output layer** — fastest, least data needed
2. **Freeze early layers, retrain later layers** — middle ground

These methods worked for CV and NLP tasks but are impractical for billion-parameter LLMs.

---

## PEFT Methods for LLMs

Modern PEFT methods add small trainable modules to a frozen model, instead of updating the entire parameter space.

### 1. LoRA — Low-Rank Adaptation ⭐ (Most Popular)

LoRA is the dominant fine-tuning method today. The key idea:

**Instead of updating the full weight matrix W (which is huge), approximate the update as a product of two small low-rank matrices: ΔW = A × B**

- The original model weights are **frozen**
- Only the small A and B matrices are trained
- At inference time, the LoRA weights are merged back: `W' = W + A×B`
- Drastically reduces the number of trainable parameters

**Why it works**: Weight updates during fine-tuning tend to have a low intrinsic rank — meaning the meaningful changes live in a low-dimensional subspace.

**Common LoRA targets**: The Q, K, V, and output projection matrices in attention layers.

### 2. QLoRA — Quantized LoRA ⭐⭐ (Even More Efficient)

QLoRA combines LoRA with **quantization**:

- The base model is loaded in **4-bit or 8-bit precision** (quantized from float32 or float16)
  - `float32 → float16 / bfloat16 → int8 → int4`
- LoRA adapters are trained in higher precision on top
- Enables fine-tuning **70B+ parameter models on a single consumer GPU**

This is the most practical method for individual researchers and small teams.

### 3. DoRA — Weight-Decomposed Low-Rank Adaptation

DoRA decomposes the pre-trained weight into **magnitude** and **direction** components, then applies LoRA-style updates to the directional component separately. Often outperforms LoRA on downstream tasks.

### 4. BitFit & IA³ (Less Common)

- **BitFit**: Only fine-tune the bias terms — extremely parameter-efficient but limited
- **IA³**: Rescales internal activations using learned vectors — very few parameters but works well for some tasks
- Both are largely superseded by LoRA/QLoRA in practice

---

## Tools for Fine-Tuning

| Tool | Best For |
|------|----------|
| **Hugging Face `transformers` + `trl`** | Full pipeline, RLHF, DPO, SFT trainer |
| **Unsloth** | Fastest LoRA/QLoRA fine-tuning; 2-5× speedup vs standard HF |

---

## LLM Ecosystem: Who Builds What

| Company | Model Family | Origin |
|---------|-------------|--------|
| Meta | Llama (llama2, llama3, llama3.1, llama3.2) | USA |
| Mistral AI | Mistral, Mixtral | France |
| OpenAI | GPT-4, GPT-4o | USA |
| Deepseek | Deepseek-V2, V3, R1 | China |
| Sarvam AI | Sarvam-1, 2 | India |

All of these go through the same three-stage pipeline: Pretraining → SFT → Preference Tuning.

---

## Course Roadmap (Upcoming Topics)

- Remaining HuggingFace concepts
- Full fine-tuning pipeline using HuggingFace
  - Instruction tuning on custom raw docs
  - Inbuilt instruction tuning datasets from HF
- Fine-tuning pipeline using **Unsloth**
- **LoRA, QLoRA, DoRA** (deep dive + assignment)
- **RLHF, DPO** (preference alignment in depth)
- **RAG & MMRAG** (Retrieval-Augmented Generation)

---

## Summary

| Stage | Method | Data | Output |
|-------|--------|------|--------|
| Pretraining | Self-supervised | Raw internet text | Base model |
| Instruction Tuning (SFT) | Supervised | IP → OP pairs | Instruct model |
| Preference Alignment | RLHF / DPO / GRPO | Human preference pairs | Aligned model |
| Custom Fine-tuning | Full FT / LoRA / QLoRA | Your domain data | Specialized model |

Fine-tuning has democratized AI — what once required a research lab with massive infrastructure can now be done by a single developer on a consumer GPU, thanks to PEFT methods like QLoRA and tools like Unsloth and Hugging Face.

---

*Notes compiled from Class 18 (23 May 2026) and Class 20 (30 May 2026)*
