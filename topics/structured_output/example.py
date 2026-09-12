from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


## Load the project key even when the shell has an older GROQ_API_KEY exported.
load_dotenv(override=True)


##  Define output structure
class TopicExplanation(BaseModel):

    definition: str = Field(
        description="Simple definition of the topic"
    )

    example: str = Field(
        description="A simple real-world example"
    )

    difficulty: str = Field(
        description="beginner, intermediate, or advanced"
    )



## call LLM
model = ChatGroq(model="openai/gpt-oss-120b",temperature=0)



## Add structured output
structured_model = model.with_structured_output(
    TopicExplanation
)




## Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful data science teacher."
    ),
    (
        "human",
        "Explain {topic} simply."
    )
])



## Create a chain that combines the prompt and structured output
chain = prompt | structured_model


## Run the chain with an input and get the output
result = chain.invoke({
    "topic": "Random Forest"
})

print(result)

print("\nDefinition:")
print(result.definition)

print("\nExample:")
print(result.example)

print("\nDifficulty:")
print(result.difficulty)
