# LangChain Learning Examples

A collection of small, runnable examples for learning LangChain with Groq. Each
topic pairs an executable Python script with study notes that explain the main
concepts and data flow.

## Topics

| Topic | What it demonstrates |
| --- | --- |
| [Prompt, model, and parser](topics/prompt_model_parser/) | Prompt templates, chat models, output parsers, LCEL chains, and `invoke()` |
| [RunnableLambda](topics/runnable_lambda/) | Adding ordinary Python functions to an LCEL pipeline |
| [Structured output](topics/structured_output/) | Returning validated Pydantic objects from an LLM |
| [Tools and agents](topics/tools/) | Defining tools and letting an agent choose and call them |
| [Conversation memory](topics/memory/) | Session-based in-memory and Redis-backed message history |
| [Simple RAG](topics/rag/simple_rag/simple_rag_explained.md) | PDF loading, chunking, embeddings, Chroma retrieval, and grounded answers |
| [Hybrid RAG](topics/rag/advance_rag/hybrid_advanced_rag_explained.md) | Dense and BM25 retrieval followed by CrossEncoder reranking |

## Project structure

```text
.
├── topics/
│   ├── prompt_model_parser/
│   │   ├── README.md
│   │   └── example.py
│   ├── runnable_lambda/
│   │   ├── README.md
│   │   └── example.py
│   ├── structured_output/
│   │   ├── README.md
│   │   └── example.py
│   ├── tools/
│   │   ├── README.md
│   │   └── example.py
│   ├── memory/
│   │   ├── README.md
│   │   ├── normal_memory/
│   │   │   ├── README.md
│   │   │   └── example.py
│   │   └── redis_memory/
│   │       ├── README.md
│   │       └── example.py
│   └── rag/
│       ├── pdf/
│       ├── simple_rag/
│       │   ├── example.py
│       │   ├── simple_rag_explained.md
│       │   └── simple-db/
│       └── advance_rag/
│           ├── example.py
│           ├── hybrid_advanced_rag_explained.md
│           └── advance-db/
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

The RAG examples use the paper in `topics/rag/pdf/` and persist their Chroma
collections in `simple-db/` and `advance-db/`.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the shared dependencies:

```bash
pip install -r requirements.txt
```

The tools and RAG examples require these additional packages:

```bash
pip install langchain langchain-chroma langchain-huggingface \
  langchain-text-splitters pypdf sentence-transformers rank-bm25
```

Create `.env` in the project root and add your Groq API key:

```dotenv
GROQ_API_KEY=your_api_key_here
```

For the Redis memory example, also start a Redis server and configure its URL:

```dotenv
REDIS_URL=redis://localhost:6379/0
```

The `.env` file is ignored by Git; do not commit credentials.

## Running the examples

Most examples can be run from the project root:

```bash
python topics/prompt_model_parser/example.py
python topics/runnable_lambda/example.py
python topics/structured_output/example.py
python topics/tools/example.py
python topics/memory/normal_memory/example.py
python topics/memory/redis_memory/example.py
```

Run each RAG example from its own directory because its PDF and persistence
paths are relative to that directory:

```bash
cd topics/rag/simple_rag
python example.py
```

```bash
cd topics/rag/advance_rag
python example.py
```

The first RAG run may take longer while the embedding and reranking models are
downloaded.
