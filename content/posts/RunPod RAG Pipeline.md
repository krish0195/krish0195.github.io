---
title: "Run Pod RAG Pipeline"
date: 2026-05-10

---
# RunPod RAG Pipeline

Complete Retrieval-Augmented Generation (RAG) system using RunPod Serverless for LLM and Embeddings.

## 🚀 Features

- ✅ RunPod Serverless LLM (vLLM) for text generation
- ✅ RunPod Infinity Embedding Worker for embeddings
- ✅ PDF document processing
- ✅ FAISS vector store with persistence
- ✅ Source attribution in answers
- ✅ LangChain orchestration
- ✅ Jupyter notebook interface

## 📁 Project Structure

```
runpod-rag/
├── data/                          # Place your PDF files here
├── vector_store/                  # Auto-created: stores FAISS index (gitignored)
├── runpod_rag_pdf.ipynb          # Main notebook with PDF support ⭐
├── embed_api_test.py              # Standalone embedding endpoint test script
└── llm_api_test.py                # Standalone LLM endpoint test script
```

## 🎯 Quick Start

### 1. Install Dependencies

```bash
pip install langchain langchain-runpod langchain-community langchain-classic faiss-cpu requests pypdf
```

### 2. Add Your PDFs

Place your PDF documents in the `data/` folder:

```bash
cp your_document.pdf data/
```

### 3. Configure API Keys

Open `runpod_rag_pdf.ipynb` and update:
- `RUNPOD_API_KEY` - Your RunPod API key
- `RUNPOD_LLM_ENDPOINT_ID` - Your vLLM endpoint ID
- `RUNPOD_EMBEDDING_ENDPOINT_ID` - Your Infinity Embedding endpoint ID

### 4. Run the Notebook

```bash
jupyter notebook runpod_rag_pdf.ipynb
```

Run all cells sequentially and start asking questions about your documents!

## 📚 Notebook

### `runpod_rag_pdf.ipynb`
**Full-featured RAG pipeline with PDF support**
- Loads PDFs from `data/` folder
- Saves vector store to disk for fast reloading
- Comprehensive comments and explanations
- Production-ready code

## 🔧 How It Works

### 1. Document Loading
- PDFs are loaded from `data/` folder using PyPDFLoader
- Each page is extracted as a separate document
- Metadata (filename, page number) is preserved

### 2. Text Chunking
- Documents are split into chunks using RecursiveCharacterTextSplitter
- Chunk size: 800 characters (configurable)
- Overlap: 150 characters to preserve context

### 3. Embedding Generation
- Each chunk is sent to RunPod Infinity Embedding worker
- Returns 384-dimensional vectors (BAAI/bge-small-en-v1.5)
- Uses OpenAI-compatible API format

### 4. Vector Storage
- FAISS index stores embeddings for fast similarity search
- Saved to `vector_store/` folder for persistence
- Loads instantly on subsequent runs (no re-embedding needed)

### 5. Question Answering
- User question is embedded using the same model
- FAISS finds top-k most similar document chunks
- Chunks + question are sent to RunPod LLM
- LLM generates answer grounded in retrieved context

## 🎨 Customization

### Adjust Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Increase for more context per chunk
    chunk_overlap=300,    # Increase to preserve more context
)
```

### Change Number of Retrieved Documents
```python
retriever=vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Retrieve top 5 instead of 3
)
```

### Modify LLM Parameters
```python
llm = ChatRunPod(
    endpoint_id=RUNPOD_LLM_ENDPOINT_ID,
    model_kwargs={
        "temperature": 0.5,    # Lower = more deterministic
        "max_tokens": 3000,    # Longer responses
    }
)
```

## 🔄 Rebuilding Vector Store

When you add new PDFs or modify existing ones:

1. Delete the `vector_store/` folder:
   ```bash
   rm -rf vector_store/
   ```

2. Re-run the notebook - it will automatically rebuild the index

Or use the rebuild cell in the vector store section of the notebook.

## 📊 Understanding Similarity Scores

When using `similarity_search_with_score()`:

- **0.0 - 0.3**: Highly relevant (very similar to query)
- **0.3 - 0.5**: Moderately relevant
- **0.5+**: Less relevant

Lower scores = better match (it's a distance metric, not similarity score)

## 🛠️ Troubleshooting

### "No PDF files found"
- Make sure PDFs are in the `data/` folder
- Check file extensions are `.pdf` (lowercase)

### "Vector store loading error"
- Delete `vector_store/` folder and rebuild
- Ensure embeddings object uses same model

### "Embedding job failed"
- Check RunPod API key is correct
- Verify embedding endpoint ID
- Check endpoint is deployed and active

### "Slow embedding process"
- Normal for first run (each chunk is embedded)
- Subsequent runs load from disk (instant)
- Consider using fewer/smaller documents for testing

## 🧪 API Test Scripts

Use these to verify your endpoints before running the full notebook:

```bash
# Test embedding endpoint
python embed_api_test.py

# Test LLM endpoint
python llm_api_test.py
```

Update the `EMBEDDING_ENDPOINT_ID`, `LLM_ENDPOINT_ID`, and `API_KEY` variables in each script before running.

## 🔑 RunPod Setup

### Required Endpoints:

1. **vLLM Endpoint** (for text generation)
   - Model: Any supported LLM (e.g., Llama, Mistral)
   - Get endpoint ID from RunPod dashboard

2. **Infinity Embedding Endpoint** (for embeddings)
   - Worker: `runpod-workers/worker-infinity-embedding`
   - Model: BAAI/bge-small-en-v1.5
   - Get endpoint ID from RunPod dashboard

### API Key:
- Get from: RunPod Dashboard → Settings → API Keys

## 📖 Additional Resources


- [Git code](https://github.com/krish0195/runpod-rag)
- [RunPod Documentation](https://docs.runpod.io/)
- [Infinity Embedding Worker](https://github.com/runpod-workers/worker-infinity-embedding)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

## 🤝 Contributing

Feel free to customize and extend this pipeline for your use case!

## 📄 License

This project is open source and available for educational and commercial use.
