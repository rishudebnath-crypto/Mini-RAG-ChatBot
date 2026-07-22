# Conversational RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, ChromaDB, Groq Llama 3.3, HuggingFace embeddings, and FastAPI.

## Features

- Conversational RAG
- Chroma Vector Database
- MultiQueryRetriever
- Persistent chat history (JSON)
- FastAPI backend
- Swagger API documentation

## Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq
- FastAPI
- Uvicorn

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Visit:

```
http://127.0.0.1:8000/docs
```

for the interactive API.
