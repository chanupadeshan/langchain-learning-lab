from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

## Load the project key even when the shell has an older GROQ_API_KEY exported.
load_dotenv(override=True)

## call LLM
model = ChatGroq(model="openai/gpt-oss-120b",temperature=0)

## format context for the prompt
## convert the retrieved documents into a single string with page numbers
def format_context(documents):
    context = ""
    for doc in documents:
        page = doc.metadata.get("page", 0) + 1
        context += (f"[Page {page}]\n"f"{doc.page_content}\n\n")
    return context.strip()

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
vectorstore = Chroma.from_documents(splits, embeddings,collection_name="pdf_rag",persist_directory="simple-db")
print("Vectorstore created.")
print("Vectorstore initialized.")

## create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever created.")

## create prompt template
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

## output parser
parser = StrOutputParser()

## create a chain that combines the prompt, model, and parser
chain = prompt | model | parser


## Run the chain with an input and get the output
def run_rag_chain(question: str):
    # Retrieve relevant documents from the vectorstore
    retrieved_docs = retriever.invoke(question)    
    # Combine the content of the retrieved documents into a single context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Invoke the chain with the question and the combined context
    result = chain.invoke({
        "context": context,
        "question": question
    })
    
    return result

print(run_rag_chain("What is the main focus of the paper?"))