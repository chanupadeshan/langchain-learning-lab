# Hybrid / Advanced RAG with Chroma, BM25, CrossEncoder, and Groq

This note explains your **hybrid RAG pipeline**.

## 1. Main Idea

Your simple RAG used only dense vector retrieval:

```text
Question
↓
Embedding Search
↓
Chroma
↓
Top Chunks
↓
LLM
```

Your advanced version uses:

```text
Dense Retrieval
+
Sparse Retrieval
+
CrossEncoder Reranking
```

Full flow:

```text
Question
   ↓
┌───────────────────┐
│                   │
Dense Search      BM25 Search
Chroma            Sparse
Embeddings        Keywords
│                   │
└─────────┬─────────┘
          ↓
      Merge Results
          ↓
    Remove Duplicates
          ↓
      CrossEncoder
       Reranking
          ↓
      Top 3 Chunks
          ↓
    Add Page Numbers
          ↓
         Prompt
          ↓
       Groq LLM
          ↓
   StrOutputParser
          ↓
        Answer
```

This is correctly described as:

> **Hybrid candidate retrieval with CrossEncoder reranking**

---

## 2. Dense Retrieval

Dense retrieval uses embeddings.

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The embedding model converts text into vectors:

```text
"Stress prediction using multimodal data"
↓
Embedding model
↓
[0.12, -0.44, 0.71, ...]
```

The vector represents the semantic meaning of the text.

Then Chroma stores those vectors:

```python
vectorstore = Chroma.from_documents(
    splits,
    embeddings,
    collection_name="pdf_rag",
    persist_directory="advance-db"
)
```

Flow:

```text
Chunks
↓
Embeddings
↓
Vectors
↓
Chroma
```

---

## 3. Dense Retriever

```python
dense_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 8}
)
```

This means:

> Retrieve the 8 chunks most semantically similar to the query.

Dense search is good at:

```text
meaning
semantic similarity
different wording with similar intent
```

Example:

```text
Question:
"What models were used for stress detection?"

Possible retrieved chunk:
"The framework evaluated BERT and DeBERTa..."
```

Even if the wording is not identical, dense search can still find it.

---

## 4. Sparse Retrieval with BM25

```python
sparse_retriever = BM25Retriever.from_documents(
    splits
)

sparse_retriever.k = 8
```

BM25 is **not an embedding method**.

It is a traditional keyword-based ranking algorithm.

BM25 is good at:

```text
exact keywords
model names
technical terms
acronyms
numbers
```

Example:

```text
Question:
"What ROC-AUC did DeBERTa-v3 achieve?"
```

BM25 is good at matching:

```text
DeBERTa-v3
ROC-AUC
```

So:

```text
Dense Retrieval
→ semantic meaning

Sparse Retrieval / BM25
→ keyword matching
```

---

## 5. What is Hybrid Retrieval?

Hybrid retrieval combines both:

```text
Dense
+
Sparse
```

Your code:

```python
dense_results = dense_retriever.invoke(query)

sparse_results = sparse_retriever.invoke(query)
```

Then:

```python
combined_results = (
    dense_results
    + sparse_results
)
```

So:

```text
Dense Results
+
BM25 Results
↓
Combined Candidate Results
```

That is the hybrid retrieval stage.

---

## 6. Why `k=8` for Both Retrievers?

You use:

```python
dense_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 8}
)

sparse_retriever.k = 8
```

So:

```text
Dense
→ up to 8 candidates

BM25
→ up to 8 candidates
```

Before duplicate removal:

```text
maximum ≈ 16 candidate chunks
```

Then the CrossEncoder evaluates those candidates and keeps the best 3.

This is better than retrieving only 3 from each source because the reranker has more candidates to compare.

---

## 7. Remove Duplicate Documents

The same chunk can appear in both:

```text
Dense Search
and
BM25 Search
```

Example:

```text
Dense:
A, B, C

BM25:
A, D, E
```

Combined:

```text
A, B, C, A, D, E
```

Chunk `A` appears twice.

Your code:

```python
unique_docs = []
seen = set()

for doc in combined_results:

    key = (
        doc.metadata.get("page", -1),
        doc.page_content
    )

    if key not in seen:

        seen.add(key)

        unique_docs.append(doc)
```

Flow:

```text
Candidate Chunk
↓
Already seen?
├── Yes → skip
└── No  → keep
```

---

## 8. Why Use `(page, page_content)` as the Key?

You create:

```python
key = (
    doc.metadata.get("page", -1),
    doc.page_content
)
```

This combines:

```text
page number
+
chunk text
```

Using only page number would be wrong because one page can have multiple different chunks.

Using both makes duplicate detection more reliable.

---

## 9. CrossEncoder Reranking

After hybrid retrieval, you have candidate chunks.

Now you ask:

> Which of these chunks is actually most relevant to the user's question?

You use:

```python
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

A CrossEncoder looks at:

```text
Question
+
Candidate Chunk
```

together and gives a relevance score.

---

## 10. Create Question-Document Pairs

```python
pairs = [
    (query, doc.page_content)
    for doc in unique_docs
]
```

Suppose:

```text
query =
"What is the main contribution?"
```

Then:

```python
pairs
```

looks like:

```python
[
    (
        "What is the main contribution?",
        "Chunk 1 text..."
    ),
    (
        "What is the main contribution?",
        "Chunk 2 text..."
    ),
    (
        "What is the main contribution?",
        "Chunk 3 text..."
    )
]
```

Each chunk is paired with the same question.

---

## 11. Score All Candidate Chunks

```python
scores = cross_encoder.predict(
    pairs
)
```

Example:

```python
scores = [
    0.91,
    0.53,
    0.84
]
```

Meaning:

```text
Chunk 1 → 0.91
Chunk 2 → 0.53
Chunk 3 → 0.84
```

Higher score generally means more relevant.

---

## 12. Pair Documents with Their Scores

```python
scored_docs = list(
    zip(
        unique_docs,
        scores
    )
)
```

Example:

```text
Documents:
doc1
doc2
doc3

Scores:
0.91
0.53
0.84
```

After `zip()`:

```python
[
    (doc1, 0.91),
    (doc2, 0.53),
    (doc3, 0.84)
]
```

So:

```text
Document + Score
↓
(document, score)
```

---

## 13. Sort Highest Score First

```python
scored_docs.sort(
    key=lambda x: x[1],
    reverse=True
)
```

Each item is:

```text
(document, score)
```

So:

```text
x[0]
→ document

x[1]
→ score
```

Therefore:

```python
key=lambda x: x[1]
```

means:

> Sort using the score.

And:

```python
reverse=True
```

means:

> Highest score first.

Example:

Before:

```python
[
    (doc1, 0.53),
    (doc2, 0.91),
    (doc3, 0.84)
]
```

After:

```python
[
    (doc2, 0.91),
    (doc3, 0.84),
    (doc1, 0.53)
]
```

---

## 14. Return the Top 3 Documents

```python
return [
    doc
    for doc, score
    in scored_docs[:3]
]
```

First:

```python
scored_docs[:3]
```

means:

> Take the first 3 highest-ranked items.

Then:

```python
for doc, score in ...
```

unpacks:

```text
(document, score)
```

And:

```python
doc
```

returns only the document.

So:

```text
Reranked candidate chunks
↓
Top 3
↓
Return Document objects only
```

---

## 15. Full Hybrid Retriever

```python
def hybrid_retriever(query):

    ## Dense search
    dense_results = dense_retriever.invoke(
        query
    )

    ## Sparse BM25 search
    sparse_results = sparse_retriever.invoke(
        query
    )

    ## Combine results
    combined_results = (
        dense_results
        + sparse_results
    )

    ## Remove duplicates
    unique_docs = []
    seen = set()

    for doc in combined_results:

        key = (
            doc.metadata.get("page", -1),
            doc.page_content
        )

        if key not in seen:

            seen.add(key)

            unique_docs.append(doc)

    ## Create CrossEncoder pairs
    pairs = [
        (query, doc.page_content)
        for doc in unique_docs
    ]

    ## Score all documents together
    scores = cross_encoder.predict(
        pairs
    )

    ## Pair document + score
    scored_docs = list(
        zip(
            unique_docs,
            scores
        )
    )

    ## Sort highest score first
    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    ## Return top 3
    return [
        doc
        for doc, score
        in scored_docs[:3]
    ]
```

---

## 16. Hybrid Retriever Mental Model

```text
Question
↓
┌────────────────────┐
│                    │
Dense Search       BM25
│                    │
↓                    ↓
8 candidates      8 candidates
│                    │
└─────────┬──────────┘
          ↓
        Merge
          ↓
 Remove Duplicates
          ↓
CrossEncoder Scoring
          ↓
Sort by Score
          ↓
       Top 3
```

---

## 17. Format Context

Your function:

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

```text
Document objects
↓
Page number + text
↓
One context string
```

Example:

```text
[Page 3]
The proposed system combines...

[Page 7]
The experimental evaluation shows...
```

This allows the LLM to mention page references.

---

## 18. Why `+ 1`?

PDF metadata often starts pages from:

```text
0
1
2
3
```

Humans use:

```text
Page 1
Page 2
Page 3
Page 4
```

So:

```python
doc.metadata.get("page", 0) + 1
```

converts zero-based page numbering into human-readable numbering.

---

## 19. Prompt

```python
prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful research-paper assistant.

        Answer the user's question using only
        the provided PDF context.

        Do not invent information.

        If the answer is not available in the
        context, say:

        "I don't know based on this PDF."

        When possible, mention the source page
        using [Page X].

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

Important variables:

```text
{context}
→ hybrid retrieved chunks

{question}
→ user's question
```

---

## 20. Output Parser

```python
parser = StrOutputParser()
```

This converts:

```text
AIMessage
↓
String
```

---

## 21. Chain

```python
chain = (
    prompt
    | model
    | parser
)
```

Flow:

```text
context + question
↓
Prompt
↓
Groq
↓
StrOutputParser
↓
Answer string
```

---

## 22. Full RAG Function

```python
def run_rag_chain(question: str):

    # Hybrid retrieval
    retrieved_docs = hybrid_retriever(
        question
    )

    # Documents → context string
    context = format_context(
        retrieved_docs
    )

    # Generate answer
    result = chain.invoke({
        "context": context,
        "question": question
    })

    return result
```

This connects:

```text
Retrieval
+
Augmentation
+
Generation
```

---

## 23. Full Question Flow

Example:

```text
"What is the main contribution of the paper?"
```

Flow:

```text
Question
↓
Dense Retrieval
↓
8 semantic candidates

+

Question
↓
BM25 Retrieval
↓
8 keyword candidates

↓
Merge both result sets
↓
Remove duplicates
↓
Create question-document pairs
↓
CrossEncoder scores each candidate
↓
Pair documents with scores
↓
Sort highest to lowest
↓
Keep top 3
↓
format_context()
↓
Prompt
↓
Groq
↓
StrOutputParser
↓
Answer
```

---

## 24. Why Use a CrossEncoder After Retrieval?

Dense/BM25 retrieval is useful for:

```text
Finding candidates quickly
```

CrossEncoder is useful for:

```text
Carefully ranking a smaller candidate set
```

So:

```text
Stage 1:
Dense + BM25
→ candidate retrieval

Stage 2:
CrossEncoder
→ precise reranking
```

This is a common two-stage retrieval design.

---

## 25. Why Not Use CrossEncoder on the Entire Database?

Imagine:

```text
10,000 chunks
```

CrossEncoding all 10,000 chunks for every question would be slow.

Instead:

```text
10,000 chunks
↓
Dense + BM25
↓
~10–16 candidates
↓
CrossEncoder
↓
Top 3
```

Much more efficient.

---

## 26. Is This RRF?

No.

Your code does:

```text
Dense Results
+
BM25 Results
↓
Merge
↓
CrossEncoder Reranking
```

It does **not** calculate Reciprocal Rank Fusion scores.

So the most accurate name is:

> **Hybrid retrieval with CrossEncoder reranking**

RRF is another possible fusion method, but it is not used in this code.

---

## 27. Simple RAG vs Hybrid RAG

### Simple RAG

```text
Question
↓
Embedding Search
↓
Chroma
↓
Top 3
↓
LLM
```

### Hybrid RAG

```text
Question
↓
┌──────────────────┐
│                  │
Dense              BM25
│                  │
└────────┬─────────┘
         ↓
       Merge
         ↓
    Deduplicate
         ↓
   CrossEncoder
         ↓
      Top 3
         ↓
        LLM
```

---

## 28. What You Have Learned

```text
PDF Loading
✅

Chunking
✅

Embeddings
✅

Chroma
✅

Dense Retrieval
✅

BM25 Sparse Retrieval
✅

Hybrid Retrieval
✅

Duplicate Removal
✅

CrossEncoder Reranking
✅

Top-k Selection
✅

Page Grounding
✅

Prompt
✅

Groq
✅

StrOutputParser
✅
```

---

## 29. Important Chroma Note

Your current code uses:

```python
vectorstore = Chroma.from_documents(
    splits,
    embeddings,
    collection_name="pdf_rag",
    persist_directory="advance-db"
)
```

This indexes the documents when the script runs.

For a cleaner real application, later separate:

```text
index.py
↓
PDF
↓
Chunks
↓
Embeddings
↓
Chroma
```

from:

```text
rag.py
↓
Load existing Chroma
↓
Hybrid retrieval
↓
Reranking
↓
Answer
```

This avoids unnecessary re-embedding every time the application starts.

---

## 30. Final Mental Model

Remember:

```text
Dense Search
→ semantic meaning

BM25
→ keyword matching

Dense + BM25
→ hybrid candidate retrieval

CrossEncoder
→ precise reranking

Top 3 chunks
→ final context

Groq
→ grounded answer
```

Final architecture:

```text
Question
   ↓
┌───────────────┐
↓               ↓
Dense           BM25
Semantic        Keyword
↓               ↓
└───────┬───────┘
        ↓
      Merge
        ↓
   Deduplicate
        ↓
  CrossEncoder
        ↓
   Best Chunks
        ↓
     Context
        ↓
      Prompt
        ↓
       Groq
        ↓
      Answer
```
