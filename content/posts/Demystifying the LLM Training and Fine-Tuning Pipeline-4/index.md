---
title: "Demystifying the LLM Training and Fine-Tuning Pipeline-part-3"
date: 2026-06-07
---
# End-to-End LLM Fine-Tuning: Non-Instruction, Instruction, and Preference Data with Hugging Face & Unsloth
---

## Table of Contents

1. [What is Fine-Tuning and Why Do We Need It?](#1-what-is-fine-tuning-and-why-do-we-need-it)
2. [The Three Types of Fine-Tuning](#2-the-three-types-of-fine-tuning)
3. [The LLM Training Pipeline (Big Picture)](#3-the-llm-training-pipeline-big-picture)
4. [Key Concepts You Must Know First](#4-key-concepts-you-must-know-first)
5. [Stage 1: Non-Instruction Fine-Tuning (Domain-Adaptive Continued Pretraining)](#5-stage-1-non-instruction-fine-tuning)
6. [Stage 2: Instruction Fine-Tuning (SFT)](#6-stage-2-instruction-fine-tuning)
7. [Stage 3: Preference Tuning (RLHF / DPO)](#7-stage-3-preference-tuning)
8. [LoRA and QLoRA — Training on a Budget](#8-lora-and-qlora--training-on-a-budget)
9. [Quantization Explained Simply](#9-quantization-explained-simply)
10. [Datasets: Built-in vs Custom](#10-datasets-built-in-vs-custom)
11. [Hugging Face Ecosystem Overview](#11-hugging-face-ecosystem-overview)
12. [Complete Code Walkthrough](#12-complete-code-walkthrough)
13. [Inference: Using Your Fine-Tuned Model](#13-inference-using-your-fine-tuned-model)
14. [Saving, Merging, and Deploying Adapters](#14-saving-merging-and-deploying-adapters)
15. [Upcoming Topics and Assignment](#15-upcoming-topics-and-assignment)

---

## 1. What is Fine-Tuning and Why Do We Need It?

A base LLM (like LLaMA, TinyLlama, GPT) is pretrained on massive general internet text. It knows language, grammar, and general facts — but it does **not** know your domain.

**Real-world use case from class:**
> "pharma → pdf → retrain or finetune my own model"

If you work in pharma, healthcare, legal, or finance, you need the model to:
- Understand domain-specific terminology (e.g., AMPK, gluconeogenesis, Metformin)
- Answer domain-specific questions correctly
- Behave in a way your users prefer

That's where fine-tuning comes in.

### Previous Topics Covered (Prerequisites)
- Basics and Fundamentals of Fine-Tuning
- Hugging Face in detail (datasets, models, tokenizers)
- Transformer internals: tokenization → token IDs → word embeddings → positional encoding → self-attention → FFNN

---

## 2. The Three Types of Fine-Tuning

```
Pretraining (Base model + General text)
        ↓
1. Non-Instruction FT  ──→  Domain knowledge (raw text: PDF, DOCX, PPT, TXT)
        ↓
2. Instruction FT      ──→  Teaches input↔output format (Q&A, summaries)
        ↓
3. Preference Tuning   ──→  Human preference: chosen vs. rejected responses
```

### Type 1: Non-Instruction Fine-Tuning
- **Data format:** Raw text (PDF, DOCX, PPT, plain text)
- **What it learns:** Domain language, terminology, writing style
- **Goal:** Make the model fluent in your domain's language
- **Also called:** Domain-Adaptive Continued Pretraining, Custom Pretraining
- **Example:** Feed all pharma research papers → model learns pharma language

### Type 2: Instruction Fine-Tuning (SFT — Supervised Fine-Tuning)
- **Data format:** Structured `instruction → input → output` pairs (JSONL/CSV)
- **What it learns:** How to respond to user questions/tasks
- **Goal:** Make the model follow instructions and answer correctly
- **Example formats:** Alpaca format, ChatGPT format (user/assistant messages)

```json
{
  "instruction": "Explain the mechanism of action of Metformin.",
  "input": "",
  "output": "Metformin primarily activates AMPK, reduces hepatic gluconeogenesis, and improves glucose uptake."
}
```

### Type 3: Preference Tuning (RLHF / DPO)
- **Data format:** `chosen` response vs. `rejected` response pairs
- **What it learns:** Which type of response humans prefer
- **Goal:** Make the model align with human values and preferences
- **Methods:** RLHF (Reinforcement Learning from Human Feedback), DPO (Direct Preference Optimization)

---

## 3. The LLM Training Pipeline (Big Picture)

From class notes — the full 10-step Non-Instruction FT pipeline:

```
Step 1:  RAW Data (from Pharma PDFs, SharePoint, Google Drive, S3, etc.)
          ↓
Step 2:  Read the Data (Data Parsing — e.g., PyMuPDF for PDFs)
          ↓
Step 3:  Cleaning of the Data (remove noise, fix hyphenation, normalize whitespace)
          ↓
Step 4:  Final Data (clean paragraphs)
          ↓
Step 5:  HF Compatible Format (Hugging Face Dataset object)
          ↓
Step 6:  Preprocessing — Tokenization (text → token IDs)
          ↓
Step 7:  Load the Model (optionally: Quantize model, add LoRA Adapter)
          ↓
Step 8:  Fine-Tune the Model using PEFT (not full model — just a subset of parameters)
          ↓
Step 9:  Save / Push to Hugging Face Hub
          ↓
Step 10: Inference → Text Prediction with trained model
```

### Where raw data comes from (real-world sources)
- SharePoint, Confluence, Google Drive
- AWS S3, Azure Blob Storage
- Internal document portals
- Download as: PDF / DOCX / HTML / TXT / Markdown

---

## 4. Key Concepts You Must Know First

### 4.1 Tokenization

Tokenization converts text into numbers the model can process.

```
"I love AI"
    ↓ tokenizer (BPE / WordPiece)
["I", "love", "AI"]
    ↓
[40, 1842, 9552]  ← token IDs
```

**Two main tokenization algorithms:**
- **BPE (Byte Pair Encoding)** — used by GPT, LLaMA
- **WordPiece** — used by BERT

**How a Transformer processes input:**
```
Token → Token IDs → Word Embedding → Positional Encoding → Self-Attention → FFNN → Output
```

### 4.2 Padding and Truncation

Training requires all sequences in a batch to be the same length.

```
Sentence 1: "My name is sunny"  →  4 tokens
Sentence 2: "My friend name is Yash and Rahul"  →  7 tokens
Sentence 3: "I am DS & AI eng and my friend is dev"  →  11 tokens
```

**Solution: Padding to MAX_LENGTH**
```
"My name is sunny" + <S> <S> <S> <S> .... <S>  →  padded to max length
```

**Smart padding approach:**
- If your 5 sentences have lengths: 413, 512, 1000, 300, 650
- Padding all to 1000 wastes GPU memory
- Instead: pad to `max_token = 512`, and **truncate** anything longer
- This creates fixed 512-token **blocks** — efficient for GPU training

### 4.3 Next Token Prediction (Causal Language Modeling)

The core task of non-instruction fine-tuning — given a sequence, predict what comes next:

```
Input:  "You need to change your setting"  (shift = 1)
Output: "need to change your setting to"
```

Each position in the input predicts the next token — this is how the model learns domain language patterns.

### 4.4 BERT vs GPT/LLaMA Architecture

| Feature | BERT | GPT / LLaMA |
|---|---|---|
| Type | Encoder | Decoder (Causal) |
| Task | Masked Language Modeling | Next Token Prediction |
| Hidden size (base) | 768 | Varies |
| Used for | Classification, NER | Generation, Chat |
| Fine-tuning type | Not used for causal FT | Used for all 3 FT types |

---

## 5. Stage 1: Non-Instruction Fine-Tuning

### Goal
Train the model on raw pharma PDFs so it learns pharma domain language, terminology, and writing patterns — **without** teaching it how to answer questions.

### Full Pipeline

```
Pharma PDF
    ↓
PDF text extraction (PyMuPDF / fitz)
    ↓
Text cleaning and normalization
    ↓
Split into paragraphs
    ↓
Hugging Face Dataset conversion
    ↓
Tokenization
    ↓
Text packing into fixed 512-token blocks
    ↓
LoRA/QLoRA fine-tuning (causal LM)
    ↓
Validate with loss curve
    ↓
Save adapter → Push to Hub
    ↓
Inference (text continuation)
```

### Step-by-Step Code

#### Install Dependencies

```python
!pip install -q -U pymupdf datasets transformers accelerate peft bitsandbytes torchao
```

#### Configuration (All Parameters in One Place)

```python
from dataclasses import dataclass, asdict

@dataclass
class Config:
    pdf_path: str = "/content/Metformin-Lipid-Therapy-Knowledge.pdf"
    model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    output_dir: str = "/content/pharma_tinyllama_lora_output"
    adapter_dir: str = "/content/pharma_tinyllama_lora_adapter"
    processed_data_dir: str = "/content/pharma_processed_data"
    min_chars_per_paragraph: int = 80
    block_size: int = 512           # tokens per training block
    test_size: float = 0.15
    seed: int = 42
    lora_r: int = 16                # LoRA rank
    lora_alpha: int = 32            # LoRA scaling
    lora_dropout: float = 0.05
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    eval_steps: int = 10
    save_steps: int = 25
    save_total_limit: int = 2
    max_steps: int = -1
```

> **Why a Config dataclass?** Keeps all hyperparameters in one place. Easy to debug, reproduce, and productionize. Use `dataclass`, `enum`, `pydantic`, or `TypedDict` — pick what fits your project.

#### Extract Text from PDF

```python
import fitz  # PyMuPDF

def extract_pdf_pages(pdf_path):
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "page": page_index,
                    "text": text,
                    "char_count": len(text),
                })
    return pages
```

#### Clean the Extracted Text

Raw PDFs have many issues — here's what we fix and why:

| Cleaning Step | What It Does | Why It Matters |
|---|---|---|
| Unicode normalization (`NFKC`) | Converts unusual chars (`ＡＭＰＫ` → `AMPK`) | Prevents tokenizer confusion |
| Remove zero-width chars (`\u200b`) | Removes invisible spaces | Prevents bad tokens |
| Remove BOM (`\ufeff`) | Removes hidden byte order marks | Keeps text clean |
| Fix hyphenated line breaks | `gluconeogene-\nsis` → `gluconeogenesis` | Prevents broken medical terms |
| Normalize spaces | Multiple spaces → single space | Consistent tokenization |
| Remove standalone page numbers | Removes lines like `1`, `23` | Avoids model learning page numbers |
| Split into paragraphs | Split on blank lines | Preserves document structure |

```python
import re
import unicodedata

def clean_pdf_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\n+", " ", paragraph)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)
```

#### Create Hugging Face Dataset

```python
from datasets import Dataset, DatasetDict

# Convert paragraph records to HF Dataset
text_dataset = Dataset.from_list(paragraph_records)

# Train/validation split
split = text_dataset.train_test_split(test_size=0.15, seed=42)
dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
```

#### Tokenize

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Common for LLaMA-style models

def tokenize_function(examples):
    return tokenizer(examples["text"])  # No padding here — handled by text packing

tokenized_datasets = dataset.map(tokenize_function, remove_columns=dataset["train"].column_names)
```

#### Text Packing (vs Padding Approach)

**Approach 1 — Padding (Simple but inefficient):**
```
Paragraph (100 tokens) + 412 padding tokens = 512 tokens  ← GPU waste!
```

**Approach 2 — Text Packing (Efficient, Industry Standard):**
```
Paragraph 1 (100 tokens) + Paragraph 2 (150 tokens) + Paragraph 3 (262 tokens) = 512 real tokens
```

No padding waste. All 512 tokens are real text. Better GPU utilization.

```python
def create_training_blocks(tokenized_examples):
    # Join all tokens into one long stream
    all_input_ids = []
    all_attention_masks = []
    for ids in tokenized_examples["input_ids"]:
        all_input_ids.extend(ids)
    for mask in tokenized_examples["attention_mask"]:
        all_attention_masks.extend(mask)

    # Cut into fixed-size blocks
    total = len(all_input_ids)
    usable = (total // config.block_size) * config.block_size

    if usable == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    all_input_ids = all_input_ids[:usable]
    all_attention_masks = all_attention_masks[:usable]

    blocks_ids, blocks_mask = [], []
    for start in range(0, usable, config.block_size):
        end = start + config.block_size
        blocks_ids.append(all_input_ids[start:end])
        blocks_mask.append(all_attention_masks[start:end])

    return {
        "input_ids": blocks_ids,
        "attention_mask": blocks_mask,
        "labels": blocks_ids.copy(),  # Labels = input_ids for causal LM
    }

final_dataset = tokenized_datasets.map(create_training_blocks, batched=True)
```

> **Key insight:** For causal LM, `labels = input_ids`. The model predicts each token from the previous ones — that's how it learns.

#### Load Model with QLoRA (4-bit Quantization)

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

use_cuda = torch.cuda.is_available()

if use_cuda:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

base_model.config.use_cache = False
```

#### Apply LoRA Adapters

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config.lora_r,             # Rank: controls adapter size
    lora_alpha=config.lora_alpha, # Scaling factor
    lora_dropout=config.lora_dropout,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: ~4M || all params: ~1.1B || trainable%: ~0.37%
```

#### Data Collator and Trainer

```python
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

# mlm=False → we are doing causal LM, not masked LM (BERT-style)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    logging_steps=1,
    eval_steps=config.eval_steps,
    save_steps=config.save_steps,
    save_total_limit=config.save_total_limit,
    fp16=use_cuda,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["validation"],
    data_collator=data_collator,
)

trainer.train()
```

> **Why `DataCollatorForLanguageModeling`?** After text packing, the collator batches multiple fixed-size blocks into tensors of shape `[batch_size, 512]`. This is the format the model expects.

---

## 6. Stage 2: Instruction Fine-Tuning

### Goal
After Stage 1, the model knows pharma language. Now teach it **how to answer questions** using structured instruction-response pairs.

### Alpaca Instruction Format

```json
{
  "instruction": "What are common side effects of Metformin?",
  "input": "",
  "output": "Common side effects include nausea, abdominal discomfort, metallic taste, and diarrhea."
}
```

### How to Get Instruction Data (Real-World Methods)

**Method 1: Human-created Q&A**
Domain experts manually write question-answer pairs. High quality but slow and expensive.

**Method 2: Existing conversations**
Use real customer support tickets, chat transcripts, doctor-patient Q&As, email threads.
Clean and convert into instruction format.

**Method 3: Synthetic data generation (LLM-assisted)**
```
Raw pharma document
    ↓
Strong LLM generates Q&A pairs automatically
    ↓
Human/domain expert reviews
    ↓
Clean instruction dataset
    ↓
Instruction fine-tuning
```
> **Important:** In regulated domains (pharma, healthcare, legal), always have domain experts review synthetic data before training.

### Format Instruction Records (Alpaca Style)

```python
def format_instruction_record(record):
    instruction = record.get("instruction", "").strip()
    input_text = record.get("input", "").strip()
    output_text = record.get("output", "").strip()

    if input_text:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output_text}"
        )
    return {"text": text}

instruction_dataset = instruction_dataset.map(format_instruction_record)
```

### Tokenization for Instruction Data (with Label Masking)

```python
def tokenize_instruction_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()

    # Mask padding positions with -100 so loss ignores them
    tokens["labels"] = [
        [t if m == 1 else -100 for t, m in zip(ids, mask)]
        for ids, mask in zip(tokens["input_ids"], tokens["attention_mask"])
    ]
    return tokens
```

> **Why -100?** PyTorch's cross-entropy loss ignores positions labeled `-100`. This ensures the model only learns from real text, not from padding tokens.

### Two Approaches for Stage 2

| Approach | Description | Best For |
|---|---|---|
| Continue same LoRA adapter | Stage 1 adapter continues training on instruction data | Simpler, recommended for learning |
| Merge Stage 1 → Add new LoRA | Merge Stage 1, then apply fresh LoRA for Stage 2 | Production, cleaner separation |

---

## 7. Stage 3: Preference Tuning

### Goal
After instruction fine-tuning, the model can answer questions. Now teach it to respond the way humans **prefer** — safer, more helpful, more accurate.

### Data Format
```
{
  "prompt": "What is a normal blood glucose level?",
  "chosen": "Fasting blood glucose below 100 mg/dL is normal.",   ← preferred
  "rejected": "It depends. Maybe 120. Maybe 200. Hard to say."    ← not preferred
}
```

### Methods
- **RLHF** (Reinforcement Learning from Human Feedback) — used by ChatGPT, Claude
- **DPO** (Direct Preference Optimization) — simpler, no separate reward model needed

---

## 8. LoRA and QLoRA — Training on a Budget

### Why Not Full Fine-Tuning?

```
LLaMA base model = 1 Billion parameters
Full fine-tuning = update all 1B parameters → needs massive GPU RAM
```

### What LoRA Does

Instead of updating the full weight matrix, LoRA **decomposes** it into two smaller matrices using **rank decomposition**:

```
Original matrix (large) → Two small matrices A and B (trainable)
   W (frozen)           →       W + A × B  (only A, B are trained)
```

The **rank (r)** controls the size of these small matrices. Lower rank = fewer parameters = less memory.

```
LLaMA 1B parameters
    ↓
LoRA adapter → only ~4M parameters (0.37% of the model!)
    ↓
90% less GPU memory
```

### QLoRA = Quantization + LoRA

Load the base model in **4-bit precision** (instead of 32-bit float), then add LoRA adapters on top. This allows fine-tuning billion-parameter models on a single consumer GPU.

```
Weight storage:
  Float32 → 4 bytes per weight → 4 GB RAM for 1B params
  4-bit   → 0.5 bytes per weight → 0.5 GB RAM for 1B params
```

### LoRA Hyperparameters Explained

| Parameter | What It Does | Typical Value |
|---|---|---|
| `r` (rank) | Size of adapter matrices. Higher = more capacity but more memory | 8–64 |
| `lora_alpha` | Scaling factor. Usually set to 2× the rank | `2 × r` |
| `lora_dropout` | Dropout inside LoRA to reduce overfitting | 0.05 |
| `target_modules` | Which layers to attach LoRA to (attention layers) | q_proj, v_proj, etc. |

### Deploying LoRA: Two Options

**Option 1 — Load adapter separately (lightweight):**
```python
from peft import PeftModel
inference_model = PeftModel.from_pretrained(base_model, config.adapter_dir)
```
Base model stays frozen; adapter loaded on top. Upload only the small adapter (~30MB).

**Option 2 — Merge permanently (standalone model):**
```python
model_with_adapter = PeftModel.from_pretrained(base_model, config.adapter_dir)
merged_model = model_with_adapter.merge_and_unload()
merged_model.save_pretrained(merged_model_dir)
```
Results in a single model file. Larger to store but easier to deploy.

---

## 9. Quantization Explained Simply

Quantization reduces the number of bits used to store each model weight.

```
Data types and their bit usage:

Float  →  32 bits  (full precision, most accurate)
int8   →   8 bits  (4× smaller)
fp16   →  16 bits  (2× smaller, commonly used for training)
4-bit  →   4 bits  (8× smaller, used in QLoRA)
```

**Practical impact:**
```
1B parameter model:
  Float32: ~4 GB RAM
  fp16:    ~2 GB RAM
  4-bit:   ~0.5 GB RAM  ← fits on a free Colab GPU!
```

**Training loop with quantization:**
```
Forward Pass (FP) → compute loss
         ↓
Backward Pass (BP) → compute gradients
         ↓
Optimizer → update only LoRA weights (A, B matrices)
```

---

## 10. Datasets: Built-in vs Custom

### Hugging Face `datasets` Library

```python
from datasets import load_dataset

# Built-in dataset (uploaded by companies, universities, open-source community)
dataset = load_dataset("alpaca", split="train")

# Custom dataset from local files
dataset = load_dataset("json", data_files="my_data.jsonl", split="train")
```

### Supported Data Formats

| Format | Best For |
|---|---|
| CSV, XLSX, TSV, JSON | Small-to-medium datasets |
| JSONL | Instruction/preference data |
| Parquet, ORC, Arrow | Large-scale / big-data training |

### Creating a Custom HF-Compatible Dataset

```python
from datasets import Dataset

records = [{"text": "Metformin is..."}, {"text": "Atorvastatin is..."}]
dataset = Dataset.from_list(records)

# Push to HF Hub so anyone can use it
dataset.push_to_hub("your-username/your-dataset-name", private=True)
```

---

## 11. Hugging Face Ecosystem Overview

### Major LLM Providers (from class)

| Company | Model |
|---|---|
| DeepSeek | DeepSeek (China) |
| Sarvam AI | Sarvam (India) |
| Mistral AI | Mistral (France) |
| OpenAI | GPT (USA) |
| Meta | LLaMA |
| Google | Gemma |

### HF AutoModel — How It Works

```
AutoModel (wrapper)
    ↓
RAW base model (weights)
    ↓ processed through
BERT (encoder) → Encoder layers → Process vectors → Prediction head
GPT/LLaMA (decoder) → Decoder layers → Next token prediction
```

BERT-Base architecture:
- 12 Transformer (encoder) layers
- 768 hidden units
- 12 attention heads
- 110M parameters
- FFNN hidden size: 3072 (4 × 768)

---

## 12. Complete Code Walkthrough

### End-to-End Summary (what each cell does)

```
Cell 9:   Config dataclass — all hyperparameters
Cell 17:  extract_pdf_pages() — PyMuPDF text extraction
Cell 25:  clean_pdf_text() — 13-step cleaning pipeline
Cell 30:  split_into_paragraph_records() — paragraph splitting
Cell 35:  Dataset.from_list() — HF Dataset creation
Cell 40:  train_test_split() — 85/15 split
Cell 42:  AutoTokenizer.from_pretrained() — load tokenizer
Cell 44:  tokenize_function() — text → token IDs
Cell 50:  create_training_blocks() — text packing into 512-token blocks
Cell 59:  AutoModelForCausalLM + BitsAndBytesConfig — 4-bit model load
Cell 60:  LoraConfig — define adapter structure
Cell 61:  get_peft_model() — attach LoRA to base model
Cell 63:  DataCollatorForLanguageModeling — batch preparation
Cell 68:  Trainer() — assemble all components
Cell 70:  trainer.train() — start training!
Cell 72:  save_pretrained() — save adapter
Cell 76:  push_to_hub() — upload to HF Hub
Cell 82:  PeftModel.from_pretrained() — reload for inference
Cell 83:  generate_completion() — text continuation inference
Cell 90:  merge_and_unload() — merge adapter permanently
```

---

## 13. Inference: Using Your Fine-Tuned Model

### Non-Instruction Inference (Text Continuation)

Since Stage 1 is trained on raw text, use **continuation-style prompts**:

```python
def generate_completion(prompt: str, max_new_tokens: int = 120) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inference_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = inference_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,    # Lower = more deterministic
            top_p=0.9,          # Nucleus sampling
            repetition_penalty=1.1,
            pad_token_id=inference_tokenizer.eos_token_id,
        )

    return inference_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompts (continuation style — NOT Q&A yet)
prompts = [
    "Metformin is one of the most widely prescribed oral antihyperglycemic agents",
    "Clinical trials have demonstrated that combining Atorvastatin with Ezetimibe",
]
```

### Instruction Inference (Q&A Style — Stage 2)

```python
def build_instruction_prompt(instruction, input_text=""):
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

# Example questions
test_questions = [
    "Explain the primary mechanism of action of metformin.",
    "Why can atorvastatin and ezetimibe reduce LDL-C more effectively together?",
    "Summarize the role of lipid nanoparticles in mRNA vaccines.",
]
```

---

## 14. Saving, Merging, and Deploying Adapters

### Saving the Adapter

```python
trainer.model.save_pretrained(config.adapter_dir)
tokenizer.save_pretrained(config.adapter_dir)
```

Files saved: `adapter_config.json`, `adapter_model.safetensors` — only ~30MB!

### Pushing to Hugging Face Hub

```python
from huggingface_hub import notebook_login
notebook_login()  # Enter your HF token

model.push_to_hub("your-username/pharma-tinyllama-lora", private=True)
tokenizer.push_to_hub("your-username/pharma-tinyllama-lora", private=True)
```

### Merging for Deployment

```python
from peft import PeftModel

# Load base model (full precision for safe merge)
base_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float16)

# Attach adapter
model_with_adapter = PeftModel.from_pretrained(base_model, config.adapter_dir)

# Merge permanently
merged_model = model_with_adapter.merge_and_unload()
merged_model.save_pretrained("/content/merged_model")
```

### Full Three-Stage Pipeline Summary

```
TinyLlama Base (1.1B params, general language)
          ↓
Stage 1 LoRA Adapter — Non-Instruction FT on pharma PDFs
    Learns: pharma terminology, drug names, domain patterns
          ↓  (merge)
Stage 1 Merged Model
          ↓
Stage 2 LoRA Adapter — Instruction FT on pharma Q&A
    Learns: how to answer pharma questions correctly
          ↓  (merge)
Stage 2 Merged Model
          ↓
Stage 3 LoRA Adapter — Preference Tuning (RLHF/DPO)
    Learns: which responses humans prefer
          ↓
Final Pharma-Tuned Model — knows domain + follows instructions + aligned with preferences
```

---

---

## Quick Reference Cheat Sheet

```
Non-Instruction FT:
  Data → raw text (PDF, DOCX)  |  Task → next token prediction  |  Output → domain language model

Instruction FT:
  Data → {instruction, input, output}  |  Task → learn to respond  |  Output → chat/QA model

Preference Tuning:
  Data → {prompt, chosen, rejected}  |  Task → learn human preference  |  Output → aligned model

LoRA: Train only small adapter matrices (A, B) → ~0.37% of parameters
QLoRA: Load model in 4-bit → add LoRA → train on consumer GPU
Quantization: float32 (4B) → fp16 (2B) → 4-bit (0.5B) per weight
Tokenization: BPE (GPT/LLaMA), WordPiece (BERT)
Text Packing: No padding waste → concatenate all tokens → split into 512-token blocks
Labels = -100: Tell PyTorch to ignore padding in loss calculation
```

---
