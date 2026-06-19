
---
title: "The Evolution of Large Language Models & Multimodal AI"
date: 2026-05-16
---

# Master Class Notes: The Evolution of Large Language Models & Multimodal AI

## 1. The Timeline and Early Architecture Evolution (2017–2018)
The journey of modern Language Models started around **2017–2018**. One of the pivotal early architectures was **ULMFiT (2017)**. 

Early-stage Model training pipelines generally followed a two-step process:
1. **Stage 1: Pretraining** (Self-Supervised/Unsupervised).
2. **Stage 2: Supervised Fine-Tuning (SFT)** (Instruction Fine-Tuning / Task-Specific).

---

## 2. Core Transformer Families
The heart of the Transformer is **self-attention**, a mechanism where the model understands each token in a sentence within the context of all other tokens in that same sentence. 

Based on this architecture, models split into three main architectural archetypes:

### 💡 Encoder-Only Models
* **Core Characteristic:** Best for deeply understanding text, but not naturally designed for generating text.
* **Key Models:** BERT, RoBERTa, DistilBERT, DeBERTa.
* **Training Objective:** Masked Language Modeling (MLM).
  * *Example:* Input: `The cat [MASK] on the mat.` $\rightarrow$ Target: `sat`. (Learns from both left and right context).
* **Primary Use Cases:** Text classification (Sentiment analysis, Named Entity Recognition, POS tagging) and Embedding generation.
  * *Example 1 (Sentiment):* `"I hate this product."` $\rightarrow$ `Negative`.
  * *Example 2 (NER):* `"Virat Kohli lives in India."` $\rightarrow$ `Virat Kohli = Person, India = Location`.
  * *Example 3 (POS Tagging):* `"Ram eats mango."` $\rightarrow$ `Ram = Noun, eats = Verb, mango = Noun`.

### 🔄 Encoder-Decoder Models
* **Core Characteristic:** Best when we need to deeply understand an input sequence and map it to a newly generated, structured output sequence.
* **Key Models:** T5 (Google), BART (Meta), mT5, FLAN-T5. *(Popularized around 2019–2020)*.
* **Primary Use Cases:** Translation, Summarization, and Question Answering.
  * *Translation Example:* Input: `"How are you?"` $\rightarrow$ Output: `"आप कैसे हैं?"`.
  * *Summarization Example:* Input: `"Transformers use self-attention to understand relationships between words and are used in modern LLMs."` $\rightarrow$ Output: `"Transformers use self-attention and power modern LLMs."`.
  * *Question Answering Example:* Context: `"BERT is an encoder-only model."` $\rightarrow$ Question: `"What type of model is BERT?"` $\rightarrow$ Output: `"Encoder-only model."`.

### ✍️ Decoder-Only Models
* **Core Characteristic:** General-purpose generative models that dominate today's LLM landscape. They predict the next token one by one in an Autoregressive (AR) fashion.
* **Key Models:** GPT family, LLaMA, Mistral, DeepSeek.
* **Training Objective:** Next Token Prediction (NTP).
  * *Example:* Input: `I love machine` $\rightarrow$ Target: `learning`.
  * *Generation loop:* It continually appends the predicted token to the input to predict the next one (e.g., `Artificial intelligence is` $\rightarrow$ `powerful` $\rightarrow$ `Artificial intelligence is powerful` $\rightarrow$ `because`).
* **Primary Use Cases:**
  * **Conversational AI:** Chatbots, QA assistants.
  * **Code Generation:** Writing Python, JavaScript, SQL, APIs.
  * **Content Writing:** Blogs, emails, scripts, summaries.
  * **Reasoning:** Step-by-step problem-solving.

---

## 3. Early vs. Modern LLM Training Pipelines

The approach to training language models has significantly scaled up over time.

| Feature | Early-Stage Pipeline (e.g., BERT/Early GPT) | Modern LLM Pipeline (e.g., GPT-4, LLaMA 3) |
| :--- | :--- | :--- |
| **Data Scale** | Smaller datasets. | Internet-scale data (measured in Petabytes/Exabytes). |
| **Parameter Scale** | Millions of parameters. | Billions to trillions of parameters. |
| **Pipeline Stages** | **2 Stages:** <br>1. Unsupervised Pretraining<br>2. SFT for specific tasks. | **3 Stages:** <br>1. Large-scale Pretraining<br>2. SFT / Instruction Tuning<br>3. Preference Alignment. |

### The 3 Stages of Modern LLM Training

1. **Stage 1: Pretraining**
   * **Data:** Unlabeled internet-scale or curated data.
   * **Goal:** Next-token prediction.
   * **What it learns:** Language structure, grammar, facts, reasoning patterns, code, and foundational world knowledge. This gives the model its baseline "general knowledge".
2. **Stage 2: Supervised Fine-Tuning (SFT) / Instruction Tuning**
   * **Data:** Human-curated instruction-response pairs (Conversational templates where humans simulate both the user and the assistant).
   * **Goal:** Transition the model from a raw text completer to a task-focused conversational assistant.
   * **What it learns:** Instruction following, response formatting, helpfulness, and task completion.
3. **Stage 3: Preference Alignment**
   * **Data:** Human or AI preference datasets (e.g., choosing which answer response is better/safer).
   * **Methods:** RLHF, DPO, ORPO, GRPO, RLAIF, Constitutional AI.
   * **What it learns:** Helpfulness, safety, safety guardrails/refusal behaviors, better reasoning, and polite tone.

[ Stage 1: Large-Scale Pretraining ]
↓
[ Stage 2: Supervised Fine-Tuning (SFT) ]
↓
[ Stage 3: Preference Alignment ]
---

## 4. Small Language Models (SLMs)
**Concept:** SLMs are highly optimized language models with fewer parameters designed to run efficiently at a lower cost and faster speed, without requiring massive cloud infrastructure.

> 💡 *Takeaway:* Not every business problem requires a massive GPT-level giant model. For specific, targeted use cases, a fine-tuned SLM is faster, significantly cheaper, and easier to control.

* **Popular SLM Families:** Microsoft Phi family, Google Gemma small variants, Qwen small variants, LLaMA small variants, and Mistral 7B style models.
* **Core Advantages:** Fewer parameters, ultra-low cost inference, lightning-fast speeds, and domain proficiency.
* **Key Use Cases:**
  * On-device/Edge AI (running locally on laptops/phones).
  * Enterprise internal chatbots.
  * Domain-specific RAG (Retrieval-Augmented Generation).
  * Fast customer support automation.
  * Private, air-gapped deployments.

---

## 5. Multimodal AI Concepts & Architectures
Multimodal LLMs accept different data types (Text, Image, Audio, Video) as inputs and can map them to diverse output types.

### Core Multimodal Foundations
* **ViT (Vision Transformer):** Instead of using traditional computer vision methods, ViT breaks down a 2D image into flattened patches (e.g., a $224 \times 224$ image divided into $16 \times 16$ patches), passes them through linear projections as patch embeddings/tokens, and processes them directly inside a standard Transformer Encoder.
* **CLIP (Contrastive Language-Image Pretraining):** Jointly trains an Image Encoder and a Text Encoder. Its training goal is to push correct image-text pair embeddings closer together in a shared semantic space and push incorrect pairs further apart. This acts as the bridge connecting vision to language.

### The Vision & Generation Toolkit
1. **ViT:** Converts images into patches/tokens.
2. **CLIP:** Maps images and text into a shared semantic space.
3. **BLIP / BLIP-2:** Used for image understanding, captioning, and Visual Question Answering (VQA).
4. **LLaVA:** Connects an image encoder directly to an LLM to allow visual instruction following.
5. **Diffusion Models:** Generates images by iteratively removing random noise step-by-step.
6. **Stable Diffusion:** Implements *Latent Diffusion* to guide image generation via a text prompt processed through a latent space.
7. **VAE (Variational Autoencoder):** Compresses high-dimensional images down into a dense, low-dimensional latent space.
8. **U-Net / DiT (Diffusion Transformer):** The underlying denoising engine. While older diffusion models relied mostly on U-Net, modern models use **DiT (Diffusion Transformers)** which replaces the U-Net architecture entirely with a Transformer framework.
9. **Cross-Attention:** The mechanism that allows text prompt embeddings to guide and shape the denoising process during image generation.

[Text Prompt] ──> [Text Encoder (CLIP/T5)] ──> [Text Embeddings]
│
▼ (Via Cross-Attention)
[Random Noise] ───────────────────────────────> [DiT / U-Net Denoiser] ──> [VAE Decoder] ──> [Final Image]

---

## 6. The Multimodal Ecosystem Landscape
A comprehensive directory of specialized models across various data modalities:

### 🖼️ Image and Vision
* **Image-to-Text / Vision Understanding:** BLIP, BLIP-2, LLaVA, Flamingo, Kosmos, GPT-4V / GPT-4o style, Gemini Vision style, Qwen-VL.
* **Text-to-Image Generation:** DALL-E, Stable Diffusion, SDXL, Imagen, Midjourney, FLUX, Firefly.
* **Image-to-Image Editing/Style Transfer:** Stable Diffusion img2img, ControlNet, InstructPix2Pix, SDEdit, DALL-E Editing, Photoshop Generative Fill, FLUX Kontext, IP-Adapter, InstantID, PhotoMaker.
  * *Use Cases:* Style transfer, inpainting, outpainting, background modifications, object removal/addition, low-to-high quality upscaling.

### 🎵 Audio and Voice
* **Audio-to-Text (ASR):** Whisper, Wav2Vec 2.0, HuBERT, DeepSpeech, Conformer.
* **Text-to-Speech (TTS) / Audio Generation:** Tacotron 2, FastSpeech, VITS, Bark, AudioLM, MusicLM, ElevenLabs-style systems.
* **Audio-to-Audio processing:** SeamlessM4T, AudioPaLM, RVC, Demucs, VoiceFixer, SpeechT5, OpenVoice, FreeVC, so-vits-svc.
  * *Use Cases:* Voice conversion/cloning, speech-to-speech translation, studio background noise removal, music vocal/instrument separation.

### 🎬 Video Generation
* **Text-to-Video Generation:** Sora, Veo, Kling, Runway Gen series, Pika, Luma Dream Machine, Stable Video Diffusion.
* **Video Understanding (Video-to-Text):** VideoBERT, VideoCLIP, Video-LLaVA, VideoChatGPT, Gemini multimodal, GPT-4o style systems.
* **Video-to-Video Editing:** Runway Gen series, Pika, Luma Dream Machine, Sora-style video editing, Stable Video Diffusion, AnimateDiff, VideoCrafter, TokenFlow, Tune-A-Video, Rerender-A-Video, EbSynth, FILM / RIFE.
  * *Use Cases:* Video style transfer (e.g., real life to anime), environment/background swap, object tracking removal, frame interpolation.

---

## 7. Developer API Ecosystem
To build applications using these architectures, developers leverage the following core cloud and model API endpoints:

### Text & LLM APIs
* OpenAI API
* Anthropic API
* Google Gemini API / Vertex AI
* AWS Bedrock
* Azure Cloud Foundry
* Inference Router Providers: Groq, OpenRouter, Together AI, Fireworks AI, Hugging Face.

### Image Generation APIs
* OpenAI
* Stability AI
* Fal.ai, Runware, Replicate, Together, Fireworks.

### Audio & Video APIs
* **Audio:** ElevenLabs, Deepgram, OpenAI, AssemblyAI, Google/Azure Speech.
* **Video:** Runway, Pika, Luma, Replicate, Fal.ai (along with specialized access paths for Google Veo and OpenAI Sora).