---
title: "Understanding Text Representation: From Encoding to Embeddings"
date: 2026-04-11
draft: false
tags: ["NLP", "Embeddings", "Machine Learning", "AI"]
---

### 🔹 Overview

In Natural Language Processing (NLP), converting text into numbers (vectors) has evolved from simple counting methods to deep semantic understanding.

This shift allows machines to move beyond keyword matching to actually understand meaning and context.

---

## 🔹 1. One-Hot Encoding (OHE)

One-Hot Encoding is the most basic form of text representation. It focuses on word presence.

Logic:
- Create a vocabulary of unique words  
- Each word → vector with one `1` and rest `0`  

Example:
"AI" → [0, 1, 0, 0]


Pros:
- Simple to implement  
- No training required  

Cons:
- Sparse vectors (memory waste)  
- No semantic meaning  
- Cannot handle new words (OOV problem)  

---

## 🔹 2. Bag of Words (BoW)

Bag of Words improves OHE by capturing word frequency.

Logic:
- Count how many times each word appears  

Example:

"AI AI ML" → [AI=2, ML=1]


Pros:
- Easy to implement  
- Works well for basic tasks (spam detection, sentiment analysis)  

Cons:
- Ignores word order  
  - "dog bites man" = "man bites dog"  
- Frequent words dominate  

---

## 🔹 3. TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF adds importance weighting to words.

Formula:
- TF (Term Frequency) → Frequency in a document  
- IDF (Inverse Document Frequency) → Rarity across documents  

IDF = log(N / DF)


Why Log?
- Prevents rare words from having extreme influence  
- Keeps model stable  

Use Cases:
- Search engines  
- Document ranking  

---

## 🔹 4. Word Embeddings (Modern NLP)

Embeddings capture semantic meaning of words.

### Word2Vec

- Introduced by Google (2013)  
- Converts words into dense vectors (e.g., 300 dimensions)  

Key Features:
- Words with similar meaning are close  
  - "movie" ≈ "film"  

- Vector arithmetic:


---

### Transformers & Contextual Embeddings

Modern models like:
- BERT  
- Sentence Transformers  
- OpenAI / Gemini APIs  

They understand context, not just words.

Example:
- "bank" → finance vs river meaning changes based on context  

---

## 🔹 Summary Comparison

| Feature | One-Hot | BoW | TF-IDF | Embeddings |
|--------|--------|-----|--------|------------|
| Goal | Word Presence | Word Count | Importance | Semantic Meaning |
| Dimensionality | High | High | High | Low |
| Sparse | Yes | Yes | Yes | No |
| Order Awareness | No | No | No | Yes |
| Context Awareness | No | No | No | Yes |

---

## 🔹 Final Takeaway

Text representation has evolved from:

➡️ Counting → Weighting → Understanding

Modern AI systems rely on embeddings to:
- Capture meaning  
- Understand context  
- Power applications like chatbots, search, and recommendation systems  

---

🚀 Mastering embeddings is essential for building real-world NLP and GenAI applications.