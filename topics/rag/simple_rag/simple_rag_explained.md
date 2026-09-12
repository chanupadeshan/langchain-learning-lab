# Simple RAG with LangChain, Chroma, Hugging Face Embeddings, and Groq

This note explains your **simple PDF RAG pipeline**.

## 1. What is RAG?

RAG means:

```text
R = Retrieval
A = Augmented
G = Generation
```

Simple flow:

```text
Question
↓
Retrieve relevant information from the PDF
↓
Add that information to the prompt
↓
LLM generates an answer
```

Without RAG:

```text
Question
↓
LLM
↓
Answer from model knowledge
```

With RAG:

```text
Question
↓
Search PDF
↓
Relevant chunks
↓
Prompt + context
↓
LLM
↓
Answer based on PDF
```

---

## 2. Full Architecture

```text
PDF
↓
PyPDFLoader
↓
Documents
↓
RecursiveCharacterTextSplitter
↓
Chunks
↓
HuggingFaceEmbeddings
↓
Vectors
↓
Chroma
↓
Retriever
↓
Top 3 relevant chunks
↓
format_context()
↓
Prompt
↓
Groq LLM
↓
StrOutputParser
↓
Final Answer
```

---

## 3. Imports

```python
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
```

What each one does:

```text
load_dotenv
→ Loads environment variables such as GROQ_API_KEY

ChatGroq
→ Connects LangChain to the Groq model

PyPDFLoader
→ Reads the PDF and creates Document objects

RecursiveCharacterTextSplitter
→ Splits large text into smaller chunks

HuggingFaceEmbeddings
→ Converts text into vectors

Chroma
→ Stores vectors and performs similarity search

ChatPromptTemplate
→ Creates the prompt

StrOutputParser
→ Converts AIMessage → normal string
```

---

## 4. Load Environment Variables

```python
load_dotenv(override=True)
```

Example `.env`:

```env
GROQ_API_KEY=your_api_key
```

`override=True` means the `.env` value can replace an older environment variable.

---

## 5. Create the LLM

```python
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
```

Purpose:

```text
Retrieved PDF context
+
User question
↓
Groq LLM
↓
Answer
```

`temperature=0` makes the model less random, which is useful for RAG.

---

## 6. Load the PDF

```python
loader = PyPDFLoader(
    "../pdf/An_Explainable_Multimodal_System_for_Stress_Assessment_and_Emotion.pdf"
)

documents = loader.load()
```

Flow:

```text
PDF
↓
PyPDFLoader
↓
Document objects
```

A LangChain `Document` mainly contains:

```python
doc.page_content
doc.metadata
```

Meaning:

```text
page_content
→ actual text

metadata
→ page number, source path, etc.
```

---

## 7. Split the PDF into Chunks

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = splitter.split_documents(
    documents
)
```

Why split?

```text
Large PDF page
↓
Chunk 1
Chunk 2
Chunk 3
...
```

Smaller chunks make retrieval more precise.

---

## 8. `chunk_size`

```python
chunk_size=1000
```

This tells the splitter to aim for chunks of roughly 1000 characters.

Example:

```text
5000-character text
↓
~1000-char chunk
~1000-char chunk
~1000-char chunk
...
```

---

## 9. `chunk_overlap`

```python
chunk_overlap=200
```

This keeps some text from the previous chunk in the next chunk.

Example:

```text
Chunk 1:
AAAAAA BBBBBB

Chunk 2:
       BBBBBB CCCCCC
       ↑ overlap
```

Why?

Because important information may sit on the boundary between two chunks.

---

## 10. Create Embeddings

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Embeddings convert text into numerical vectors.

Example:

```text
"Stress assessment using multimodal signals"
↓
Embedding model
↓
[0.12, -0.41, 0.77, ...]
```

Important:

```text
Groq LLM
→ generates answers

Embedding model
→ converts text into vectors
```

They do different jobs.

---

## 11. Create Chroma Vector Store

```python
vectorstore = Chroma.from_documents(
    splits,
    embeddings,
    collection_name="pdf_rag",
    persist_directory="simple-db"
)
```

Conceptually:

```text
Chunk 1 → Embedding → Vector
Chunk 2 → Embedding → Vector
Chunk 3 → Embedding → Vector
                         ↓
                       Chroma
```

Because you use:

```python
persist_directory="simple-db"
```

Chroma saves the database to disk.

---

## 12. Why Use a Vector Store?

A vector store helps with **semantic search**.

Example question:

```text
"What is the main contribution?"
```

The PDF may say:

```text
"The proposed framework introduces..."
```

Even though the exact words are different, embedding similarity can still find that chunk.

So:

```text
Keyword matching
→ exact/similar words

Vector search
→ similar meaning
```

---

## 13. Create the Retriever

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
```

`k=3` means:

> Return the 3 most relevant chunks.

Flow:

```text
Question
↓
Question embedding
↓
Compare with stored chunk vectors
↓
Similarity search
↓
Top 3 chunks
```

---

## 14. Retrieval

Inside your RAG function:

```python
retrieved_docs = retriever.invoke(
    question
)
```

This is the **Retrieval** step.

Example:

```text
Question
↓
Retriever
↓
Document 4
Document 9
Document 15
```

These returned values are still `Document` objects.

---

## 15. Your `format_context()` Function

You wrote:

```python
def format_context(documents):
    context = ""

    for doc in documents:

        page = (
            doc.metadata.get("page", 0)
            + 1
        )

        context += (
            f"[Page {page}]\n"
            f"{doc.page_content}\n\n"
        )

    return context.strip()
```

Purpose:

> Convert retrieved `Document` objects into one text string while keeping page numbers.

Example input:

```text
Document 1 → page 2
Document 2 → page 5
```

Output:

```text
[Page 3]
First retrieved chunk...

[Page 6]
Second retrieved chunk...
```

Why `+ 1`?

PDF metadata often uses:

```text
0, 1, 2, 3...
```

Humans use:

```text
Page 1, Page 2, Page 3...
```

So:

```python
page = doc.metadata.get("page", 0) + 1
```

converts zero-based numbering to human-readable numbering.

---

## 16. Small Issue in Your Current Code

You define:

```python
format_context()
```

but your current `run_rag_chain()` uses:

```python
context = "\n\n".join(
    [doc.page_content for doc in retrieved_docs]
)
```

That removes page numbers.

If you want page references, use:

```python
context = format_context(
    retrieved_docs
)
```

Recommended function:

```python
def run_rag_chain(question: str):

    retrieved_docs = retriever.invoke(
        question
    )

    context = format_context(
        retrieved_docs
    )

    result = chain.invoke({
        "context": context,
        "question": question
    })

    return result
```

---

## 17. Create the Prompt

```python
prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful AI assistant.

        Answer the user's question using only
        the provided PDF context.

        If the answer is not in the context,
        say: I don't know based on this PDF.

        Context:
        {context}
        """
    ),

    (
        "human",
        "{question}"
    )
])
```

The prompt has two variables:

```text
{context}
→ retrieved PDF chunks

{question}
→ user question
```

Example final prompt:

```text
SYSTEM:

Context:
[Page 3]
The proposed system...

[Page 8]
Experimental results...

USER:

What is the main focus of the paper?
```

---

## 18. Why Tell the LLM to Use Only the Context?

This instruction:

```text
Answer using only the provided PDF context.
```

helps reduce hallucination.

The fallback:

```text
I don't know based on this PDF.
```

means the model should not invent an answer when the retrieved evidence does not contain it.

---

## 19. Output Parser

```python
parser = StrOutputParser()
```

The model normally returns:

```text
AIMessage
```

The parser converts:

```text
AIMessage
↓
String
```

Example:

```text
AIMessage(content="The paper focuses on...")
↓
StrOutputParser
↓
"The paper focuses on..."
```

---

## 20. Create the Chain

```python
chain = (
    prompt
    | model
    | parser
)
```

This is a fixed LCEL pipeline:

```text
Input dictionary
↓
Prompt
↓
Groq LLM
↓
StrOutputParser
↓
String
```

The input is:

```python
{
    "context": context,
    "question": question
}
```

---

## 21. The RAG Function

Recommended version:

```python
def run_rag_chain(question: str):

    # Step 1: Retrieve relevant chunks
    retrieved_docs = retriever.invoke(
        question
    )

    # Step 2: Documents → context string
    context = format_context(
        retrieved_docs
    )

    # Step 3: Generate answer
    result = chain.invoke({
        "context": context,
        "question": question
    })

    return result
```

---

## 22. What Happens When You Ask a Question?

Question:

```text
What is the main focus of the paper?
```

Flow:

```text
Question
↓
Retriever
↓
Top 3 relevant chunks
↓
format_context()
↓
One context string
↓
ChatPromptTemplate
↓
Groq
↓
StrOutputParser
↓
Answer
```

---

## 23. Two Main RAG Phases

A RAG system has two important phases.

### Phase 1 — Indexing

```text
PDF
↓
Load
↓
Split
↓
Embed
↓
Store in Chroma
```

This prepares the knowledge base.

### Phase 2 — Retrieval + Generation

```text
Question
↓
Retrieve
↓
Relevant chunks
↓
Prompt
↓
LLM
↓
Answer
```

---

## 24. Why Not Send the Entire PDF?

Sending the entire PDF can cause:

```text
More tokens
More irrelevant information
Higher cost
Context-window problems
Less focused answers
```

RAG instead does:

```text
Question
↓
Only retrieve relevant chunks
↓
Send small useful context
↓
Generate answer
```

---

## 25. Simple RAG vs Hybrid RAG

Your simple RAG uses:

```text
Dense retrieval only
↓
Embedding search
↓
Chroma
```

Your advanced hybrid RAG uses:

```text
Dense retrieval
+
BM25 sparse retrieval
+
CrossEncoder reranking
```

So:

```text
Simple RAG
→ Embeddings + Chroma

Hybrid RAG
→ Embeddings + Chroma + BM25 + Reranker
```

---

## 26. Core Components to Remember

```text
Loader
→ reads PDF

Document
→ text + metadata

Splitter
→ makes chunks

Embedding Model
→ text → vector

Chroma
→ stores vectors

Retriever
→ searches for relevant chunks

format_context()
→ Documents → string

Prompt
→ combines instructions + context + question

Groq
→ generates answer

StrOutputParser
→ AIMessage → string
```

---

## 27. Clean Version of Your Code

```python
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma


load_dotenv(override=True)


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def format_context(documents):

    context = ""

    for doc in documents:

        page = (
            doc.metadata.get("page", 0)
            + 1
        )

        context += (
            f"[Page {page}]\n"
            f"{doc.page_content}\n\n"
        )

    return context.strip()


loader = PyPDFLoader(
    "../pdf/An_Explainable_Multimodal_System_for_Stress_Assessment_and_Emotion.pdf"
)

documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = splitter.split_documents(
    documents
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="pdf_rag",
    persist_directory="simple-db"
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful AI assistant.

        Answer the user's question using only
        the provided PDF context.

        If the answer is not in the context,
        say:
        I don't know based on this PDF.

        When possible, mention the source page.

        Context:
        {context}
        """
    ),

    (
        "human",
        "{question}"
    )
])


parser = StrOutputParser()


chain = (
    prompt
    | model
    | parser
)


def run_rag_chain(question: str):

    retrieved_docs = retriever.invoke(
        question
    )

    context = format_context(
        retrieved_docs
    )

    result = chain.invoke({
        "context": context,
        "question": question
    })

    return result


print(
    run_rag_chain(
        "What is the main focus of the paper?"
    )
)
```

---

## 28. Final Mental Model

Remember:

```text
KNOWLEDGE PREPARATION

PDF
↓
Documents
↓
Chunks
↓
Embeddings
↓
Chroma
```

Then:

```text
QUESTION ANSWERING

Question
↓
Retriever
↓
Relevant Chunks
↓
Context
↓
Prompt
↓
LLM
↓
Answer
```

And finally:

```text
Retrieval
+
Augmented Context
+
Generation
=
RAG
```
