from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent
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

## Create tools
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def subtract_numbers(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


## create agent
agent = create_agent(
    model=model,
    tools=[
        add_numbers,
        multiply_numbers,
        subtract_numbers,
    ],
    system_prompt="""You are a helpful AI assistant.

                  Use the available tools when calculation is needed.""")

## Run the agent
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Multiply 20 by 5, then add 10."
        }
    ]
})

print(result)
print("--------------------")
print("Agent's response:", result["messages"][-1].content)