from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


## Load the project key even when the shell has an older GROQ_API_KEY exported.
load_dotenv(override=True)

## call LLM
model = ChatGroq(model="openai/gpt-oss-120b",temperature=0)


## Load and process the PDF
loader = PyPDFLoader("../pdf/An_Explainable_Multimodal_System_for_Stress_Assessment_and_Emotion.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} documents from PDF.")


## Split the documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(documents)
print(f"Split into {len(splits)} chunks.")

## create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## create vectorstore
vectorstore = Chroma.from_documents(splits, embeddings,collection_name="pdf_rag",persist_directory="advance-db")
print("Vectorstore created.")
print("Vectorstore initialized.")

## create dence retriever
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
print("Dense retriever created.")

## create sparse retriever
sparse_retriever = BM25Retriever.from_documents(splits)
print("Sparse retriever created.")

sparse_retriever.k = 8
print("Sparse retriever initialized with k=8.")

## cross encoder for re-ranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross encoder created.")

## hybrid retriever
def hybrid_retriever(query):

    ## Dense search
    dense_results = dense_retriever.invoke(query)

    ## Sparse BM25 search
    sparse_results = sparse_retriever.invoke(query)

    ## Combine results
    combined_results = (dense_results + sparse_results)

    ## Remove duplicates
    unique_docs = []
    seen = set()

    for doc in combined_results:
        key = (doc.metadata.get("page", -1),doc.page_content)

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)


    ## Create CrossEncoder pairs
    pairs = [(query,doc.page_content) for doc in unique_docs]


    ## Score all documents together
    scores = cross_encoder.predict(pairs)


    ## Combine document + score
    scored_docs = list(zip(unique_docs,scores))


    ## Sort highest score first
    scored_docs.sort(key=lambda x: x[1],reverse=True)


    ## Return top 3 documents
    return [doc for doc, score in scored_docs[:3]]



## format context for the prompt
## convert the retrieved documents into a single string with page numbers
def format_context(documents):
    context = ""
    for doc in documents:
        page = doc.metadata.get("page", 0) + 1
        context += (f"[Page {page}]\n"f"{doc.page_content}\n\n")
    return context.strip()


## create prompt template
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

## output parser
parser = StrOutputParser()

## create a chain that combines the prompt, model, and parser
chain = prompt | model | parser

## Run the chain with an input and get the output
def run_rag_chain(question: str):
    # Retrieve relevant documents from the hybrid retriever
    retrieved_docs = hybrid_retriever(question)
    
    # Format the context for the prompt
    context = format_context(retrieved_docs)
    
    # Invoke the chain with the question and the formatted context
    result = chain.invoke({
        "context": context,
        "question": question
    })
    
    return result 

print(run_rag_chain("What is the main contribution of the paper?"))