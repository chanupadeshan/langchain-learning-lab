import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import (ChatPromptTemplate,MessagesPlaceholder)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (RedisChatMessageHistory)


## Load the project key even when the shell has an older GROQ_API_KEY exported.
load_dotenv(override=True)

## call LLM
model = ChatGroq(model="openai/gpt-oss-120b",temperature=0)



## Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

## Create output parser
parser = StrOutputParser()

## Create a chain that combines the prompt, model, and parser
chain = prompt | model | parser


redis_url = os.getenv("REDIS_URL")
if not redis_url:
    raise ValueError("REDIS_URL is missing from the .env file")

def get_memory(session_id):
    return RedisChatMessageHistory(session_id=session_id,url=redis_url)

## add memory to the chain
chain_with_memory = RunnableWithMessageHistory(
    chain,get_memory,input_messages_key="input",history_messages_key="history")

## session id
config = {
    "configurable":{
        "session_id":"1234"
    }
}

## first message
result = chain_with_memory.invoke({
    "input": "My name is Chanupa."
}, config)

print("AI:", result)

## second message
result = chain_with_memory.invoke({
    "input": "What is my name?"
}, config)  

print("AI:", result)
