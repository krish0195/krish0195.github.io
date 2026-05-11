---

title: "Understanding Text Representation: From Encoding to Embeddings"
date: 2026-04-18
draft: false
tags: ["NLP", "Embeddings", "Machine Learning", "AI", "Word2Vec"]
-----------------------------------------------------------------

## 🔹 Overview

In Natural Language Processing (NLP), converting text into numerical form (vectors) has evolved from simple counting techniques to deep semantic understanding.

This transformation enables machines to move beyond keyword matching and truly understand meaning and context.

---

## 🔹 Types of Text Representation

### 1. One-Hot Encoding (OHE)

One-Hot Encoding is the most basic form of text representation. It focuses on word presence.

**Logic:**

* Create a vocabulary of unique words
* Each word → vector with one `1` and rest `0`

**Example:**

```
AI → [0, 1, 0, 0]
```

**Pros:**

* Simple to implement
* No training required

**Cons:**

* Sparse vectors (memory inefficient)
* No semantic meaning
* Cannot handle new words (OOV problem)

---

### 2. Bag of Words (BoW)

Bag of Words improves OHE by capturing word frequency.

**Logic:**

* Count how many times each word appears

**Example:**

```
"AI AI ML" → [AI=2, ML=1]
```

**Pros:**

* Easy to implement
* Works well for basic tasks (spam detection, sentiment analysis)

**Cons:**

* Ignores word order

  * "dog bites man" = "man bites dog"
* Frequent words dominate

---

### 3. TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF assigns importance weights to words.

**Formula:**

* TF → Frequency in a document
* IDF → Rarity across documents

IDF = log(N / DF)

**Why Log?**

* Prevents rare words from having extreme influence
* Keeps model stable

**Use Cases:**

* Search engines
* Document ranking

---

## 🔹 Example Dataset (From Notes)

```
D1: people watch movie  
D2: people watch cricket  
D3: people like movie  
D4: people like cricket  
```

### Vocabulary

```
{people, watch, like, movie, cricket}
```

---

## 🔹 One-Hot Encoding (Document Representation)

| Document | Vector          |
| -------- | --------------- |
| D1       | [1, 1, 0, 1, 0] |
| D2       | [1, 1, 0, 0, 1] |
| D3       | [1, 0, 1, 1, 0] |
| D4       | [1, 0, 1, 0, 1] |

---

## 🔹 Word-Level Representation

### D1: "people watch movie"

```
people  → [1, 0, 0, 0, 0]
watch   → [0, 1, 0, 0, 0]
movie   → [0, 0, 0, 1, 0]
```

### D2: "people watch cricket"

```
people  → [1, 0, 0, 0, 0]
watch   → [0, 1, 0, 0, 0]
cricket → [0, 0, 0, 0, 1]
```

---

## 🔹 Limitations of One-Hot Encoding

* High dimensional vectors
* No semantic relationships
* Similar words are treated as completely different

---

## 🔹 4. Word Embeddings (Modern NLP)

Embeddings capture semantic meaning of words and overcome limitations of earlier methods.

---

### 🔸 Word2Vec

* Introduced by Google (2013)
* Converts words into dense vectors (e.g., 100–300 dimensions)

**Key Features:**

* Words with similar meanings are close

  * "movie" ≈ "film"

* Learns relationships from context

**Approaches:**

* CBOW (Continuous Bag of Words) → Predict word from context
* Skip-Gram → Predict context from word

---

### 🔸 Transformers & Contextual Embeddings

Modern models include:

* BERT
* Sentence Transformers
* OpenAI / Gemini APIs

They understand **context**, not just words.

**Example:**

* "bank" → financial institution vs river bank (based on context)

---

## 🔹 Summary Comparison

| Feature           | One-Hot       | BoW        | TF-IDF     | Embeddings       |
| ----------------- | ------------- | ---------- | ---------- | ---------------- |
| Goal              | Word Presence | Word Count | Importance | Semantic Meaning |
| Dimensionality    | High          | High       | High       | Low              |
| Sparse            | Yes           | Yes        | Yes        | No               |
| Order Awareness   | No            | No         | No         | Yes              |
| Context Awareness | No            | No         | No         | Yes              |

---

## 🔹 Final Takeaway

Text representation has evolved from:

➡️ Counting → Weighting → Understanding

Modern AI systems rely on embeddings to:

* Capture meaning
* Understand context
* Enable applications like chatbots, search engines, and recommendation systems

---

## 🚀 Conclusion

If you're building modern NLP or GenAI applications:

* Avoid relying only on BoW / TF-IDF
* Move toward embeddings like:

  * Word2Vec
  * GloVe
  * Transformers

👉 These techniques power today's intelligent systems.

---

💡 **Key Insight:**
Mastering embeddings is essential for real-world AI applications.
