from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

## Load the project key even when the shell has an older GROQ_API_KEY exported.
load_dotenv(override=True)

## call LLM
model = ChatGroq(model="openai/gpt-oss-120b",temperature=0)


## Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Teacher."),
    ("human", "Explain {input} simply.")
])

## Create output parser
parser = StrOutputParser()

## python function
def make_uppercase(text: str) -> str:
    return text.upper()

## convert function to a RunnableLambda
uppercase_runnable = RunnableLambda(make_uppercase)

## create a chain that compines the prompt, model, parser, and RunnableLambda
chain = prompt | model | parser | uppercase_runnable

# Run
result = chain.invoke({
    "input": "machine learning"
})


print(result)
