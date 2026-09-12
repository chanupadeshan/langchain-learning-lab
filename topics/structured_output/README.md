# LangChain Structured Output — Study Note

This note explains structured output in LangChain using Pydantic, `BaseModel`, `Field`, `ChatGroq`, `ChatPromptTemplate`, and `with_structured_output()`.

---

## 1. What is structured output?

Normally, an LLM returns free-form text.

```text
Random Forest is an ensemble learning algorithm.
It combines multiple decision trees.
It is beginner-friendly.
```

Sometimes your Python program needs predictable fields instead:

```python
{
    "definition": "...",
    "example": "...",
    "difficulty": "beginner"
}
```

That is **structured output**.

```text
Normal LLM
   ↓
Free-form text

Structured LLM
   ↓
Defined schema
   ↓
Predictable Python object
```

---

## 2. Full example

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


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


model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


structured_model = model.with_structured_output(
    TopicExplanation
)


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


chain = prompt | structured_model


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
```

---

## 3. `BaseModel`

Import:

```python
from pydantic import BaseModel
```

`BaseModel` comes from Pydantic.

It lets us define the structure of the output.

```python
class TopicExplanation(BaseModel):
    definition: str
    example: str
    difficulty: str
```

This means the final output should contain:

```text
TopicExplanation
├── definition : string
├── example    : string
└── difficulty : string
```

Think:

```text
BaseModel = output structure
```

---

## 4. Why use `BaseModel`?

Without a schema, the LLM may return different formats each time.

With a schema, we tell the model:

```text
I want these exact fields.
```

For example:

```python
class TopicExplanation(BaseModel):
    definition: str
    example: str
    difficulty: str
```

This gives your application predictable data.

---

## 5. `Field`

Import:

```python
from pydantic import Field
```

Example:

```python
definition: str = Field(
    description="Simple definition of the topic"
)
```

`Field` gives extra information about a field.

The description helps explain what value belongs there.

Example:

```python
difficulty: str = Field(
    description="beginner, intermediate, or advanced"
)
```

Think:

```text
Field name
   ↓
difficulty

Field description
   ↓
What should go inside difficulty?
```

---

## 6. Create the normal model

```python
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
```

At this stage, this is a normal chat model.

Conceptually:

```text
model
 ↓
AIMessage / normal model output
```

---

## 7. `with_structured_output()`

This is the main concept.

```python
structured_model = model.with_structured_output(
    TopicExplanation
)
```

This tells the model:

```text
Return output that matches TopicExplanation.
```

Conceptually:

```text
ChatGroq
   +
TopicExplanation schema
   ↓
Structured model
```

---

## 8. Prompt

```python
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
```

Input:

```python
{
    "topic": "Random Forest"
}
```

becomes:

```text
System:
You are a helpful data science teacher.

Human:
Explain Random Forest simply.
```

---

## 9. Chain

Previously you used:

```python
chain = prompt | model | StrOutputParser()
```

That gives plain text.

With structured output:

```python
chain = prompt | structured_model
```

You do not need `StrOutputParser()` here because the structured model already controls the output format.

---

## 10. Normal output vs structured output

### Normal model

```python
chain = prompt | model
```

Output:

```text
AIMessage
```

### Model + `StrOutputParser`

```python
chain = prompt | model | StrOutputParser()
```

Output:

```text
String
```

### Structured model

```python
chain = prompt | structured_model
```

Output:

```text
TopicExplanation object
```

---

## 11. `invoke()`

```python
result = chain.invoke({
    "topic": "Random Forest"
})
```

Flow:

```text
{"topic": "Random Forest"}
          ↓
ChatPromptTemplate
          ↓
Structured ChatGroq model
          ↓
TopicExplanation object
```

---

## 12. Accessing fields

Because `result` is a structured object:

```python
result.definition
result.example
result.difficulty
```

Example:

```python
print(result.definition)
```

You do not need to manually parse the text.

---

## 13. Example result

Conceptually:

```python
TopicExplanation(
    definition="Random Forest is an ensemble machine learning algorithm.",
    example="It can be used to predict whether a customer will buy a product.",
    difficulty="beginner"
)
```

Then:

```python
result.definition
```

returns:

```text
Random Forest is an ensemble machine learning algorithm.
```

---

## 14. Why is structured output useful?

It is useful when building:

```text
APIs
RAG systems
Agents
Classification systems
Data extraction systems
Dashboards
Automation workflows
Backend applications
```

Instead of manually extracting values from a paragraph, your program gets fields directly.

---

## 15. Example — sentiment analysis

```python
class SentimentResult(BaseModel):

    sentiment: str

    confidence: float

    reason: str
```

Possible output:

```python
SentimentResult(
    sentiment="positive",
    confidence=0.94,
    reason="The user expresses satisfaction."
)
```

You can then use:

```python
result.sentiment
result.confidence
result.reason
```

---

## 16. Other field types

You can use more than `str`.

```python
class StudentResult(BaseModel):

    name: str

    age: int

    score: float

    passed: bool

    subjects: list[str]
```

Possible result:

```python
StudentResult(
    name="Kamal",
    age=22,
    score=87.5,
    passed=True,
    subjects=["Machine Learning", "Deep Learning"]
)
```

---

## 17. Restrict values with `Literal`

Sometimes a field should allow only specific values.

```python
from typing import Literal


class TopicExplanation(BaseModel):

    definition: str

    example: str

    difficulty: Literal[
        "beginner",
        "intermediate",
        "advanced"
    ]
```

Now `difficulty` is restricted to:

```text
beginner
intermediate
advanced
```

---

## 18. Why structured output is useful in RAG

Later, your RAG system can return:

```python
class RAGAnswer(BaseModel):

    answer: str

    source_page: int

    confidence: float
```

Then your code can use:

```python
result.answer
result.source_page
result.confidence
```

instead of trying to extract these values from one long string.

---

## 19. Main difference

### Plain text

```python
chain = prompt | model | StrOutputParser()
```

Returns:

```text
String
```

Use when:

```text
You only need readable text.
```

### Structured output

```python
structured_model = model.with_structured_output(
    TopicExplanation
)

chain = prompt | structured_model
```

Returns:

```text
Structured Python object
```

Use when:

```text
Your program needs predictable fields.
```

---

## 20. Quick summary

### `BaseModel`

```text
Defines the output structure.
```

### `Field`

```text
Describes what each field should contain.
```

### `with_structured_output()`

```text
Makes the model return output matching the schema.
```

### `result.definition`

```text
Access the definition field.
```

### `result.example`

```text
Access the example field.
```

### `result.difficulty`

```text
Access the difficulty field.
```

---

## 21. Most important thing to remember

Normal text chain:

```python
chain = prompt | model | StrOutputParser()
```

```text
Prompt
 ↓
Model
 ↓
String
```

Structured chain:

```python
structured_model = model.with_structured_output(
    TopicExplanation
)

chain = prompt | structured_model
```

```text
Prompt
 ↓
Structured Model
 ↓
Pydantic Object
```

Final mental model:

```text
BaseModel
   ↓
Defines schema
   ↓
with_structured_output()
   ↓
Applies schema to model
   ↓
Prompt | Structured Model
   ↓
invoke()
   ↓
Structured Python Object
```
