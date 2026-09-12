from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


## Load the project key even when the shell has an older GROQ_API_KEY exported.
load_dotenv(override=True)

## call LLM
model = ChatGroq(model="openai/gpt-oss-120b",temperature=0)

## Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}")
])


## Create output parser
parser = StrOutputParser()

## Create a chain that combines the prompt, model, and parser
chain = prompt | model | parser

## Run the chain with an input and get the output
result = chain.invoke({
    "input": "Write a short poem about nature."
})


print(result)
