# LangChain `RunnableLambda` — Study Note

This note explains the LangChain `RunnableLambda` concept.

---

# 1. What is `RunnableLambda`?

Import:

```python
from langchain_core.runnables import RunnableLambda
```

`RunnableLambda` lets you take a normal Python function and use it inside a LangChain chain.

Think:

```text
Normal Python Function
        ↓
RunnableLambda
        ↓
LangChain Runnable
```

Example:

```python
def make_uppercase(text):
    return text.upper()


uppercase = RunnableLambda(make_uppercase)
```

Now:

```python
uppercase
```

can be used inside a LangChain chain.

---

# 2. Why do we need it?

Normally, LangChain chains contain things like:

```text
Prompt
Model
Parser
Retriever
```

But sometimes you also want your own Python logic.

For example:

```text
Clean text
Convert text to uppercase
Extract a value
Format retrieved documents
Modify a dictionary
Calculate something
```

A normal Python function cannot always be directly inserted into an LCEL chain.

`RunnableLambda` wraps the function so LangChain can run it like any other runnable.

---

# 3. Basic Example

```python
from langchain_core.runnables import RunnableLambda


def make_uppercase(text):
    return text.upper()


uppercase = RunnableLambda(make_uppercase)


result = uppercase.invoke(
    "hello langchain"
)


print(result)
```

Output:

```text
HELLO LANGCHAIN
```

Flow:

```text
"hello langchain"
        ↓
RunnableLambda
        ↓
make_uppercase()
        ↓
"HELLO LANGCHAIN"
```

---

# 4. Use `RunnableLambda` Inside a Chain

Example:

```python
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


load_dotenv()


model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "Explain {topic} simply.")
])


parser = StrOutputParser()


def make_uppercase(text):
    return text.upper()


uppercase = RunnableLambda(
    make_uppercase
)


chain = (
    prompt
    | model
    | parser
    | uppercase
)


result = chain.invoke({
    "topic": "machine learning"
})


print(result)
```

---

# 5. Chain Flow

```text
Input Dictionary
      ↓
Prompt
      ↓
Model
      ↓
AIMessage
      ↓
StrOutputParser
      ↓
String
      ↓
RunnableLambda
      ↓
Python Function
      ↓
Final String
```

For our example:

```text
{"topic": "machine learning"}
        ↓
Prompt
        ↓
Groq Model
        ↓
Parser
        ↓
"Machine learning is..."
        ↓
make_uppercase()
        ↓
"MACHINE LEARNING IS..."
```

---

# 6. Important Point About Order

The input type must match what your Python function expects.

Example:

```python
def make_uppercase(text):
    return text.upper()
```

This function expects:

```text
string
```

So it makes sense to place it after:

```python
StrOutputParser()
```

because:

```python
StrOutputParser()
```

returns a string.

Correct:

```python
prompt | model | parser | uppercase
```

Flow:

```text
Prompt
 ↓
Model
 ↓
AIMessage
 ↓
Parser
 ↓
String
 ↓
RunnableLambda
```

---

# 7. Why This Could Fail

Imagine you write:

```python
chain = prompt | model | uppercase
```

But:

```python
model
```

returns an `AIMessage`.

Your function:

```python
def make_uppercase(text):
    return text.upper()
```

expects a string.

So:

```text
AIMessage
   ↓
make_uppercase()
```

may fail.

You would either:

```python
prompt | model | StrOutputParser() | uppercase
```

or make your function handle an `AIMessage`.

---

# 8. `RunnableLambda` with a Dictionary

A `RunnableLambda` can also receive dictionaries.

Example:

```python
from langchain_core.runnables import RunnableLambda


def calculate_total(data):

    price = data["price"]
    quantity = data["quantity"]

    return price * quantity


calculate = RunnableLambda(
    calculate_total
)


result = calculate.invoke({
    "price": 100,
    "quantity": 5
})


print(result)
```

Output:

```text
500
```

Flow:

```text
{
  "price": 100,
  "quantity": 5
}
        ↓
RunnableLambda
        ↓
calculate_total()
        ↓
500
```

---

# 9. `RunnableLambda` Can Change the Shape of Data

Example:

```python
def prepare_data(data):

    return {
        "topic": data["topic"],
        "level": "beginner"
    }


prepare = RunnableLambda(
    prepare_data
)
```

Input:

```python
{
    "topic": "Random Forest"
}
```

Output:

```python
{
    "topic": "Random Forest",
    "level": "beginner"
}
```

This is useful when one step needs a different input format from the previous step.

---

# 10. Example with a Prompt

```python
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


load_dotenv()


model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


def prepare_input(data):

    return {
        "topic": data["topic"],
        "level": "beginner"
    }


prepare = RunnableLambda(
    prepare_input
)


prompt = ChatPromptTemplate.from_template(
    "Explain {topic} for a {level} student."
)


chain = (
    prepare
    | prompt
    | model
    | StrOutputParser()
)


result = chain.invoke({
    "topic": "neural networks"
})


print(result)
```

Flow:

```text
{"topic": "neural networks"}
        ↓
RunnableLambda
        ↓
{
  "topic": "neural networks",
  "level": "beginner"
}
        ↓
Prompt
        ↓
Model
        ↓
Parser
        ↓
String
```

---

# 11. Inline Lambda Function

You do not always need a named Python function.

You can write:

```python
uppercase = RunnableLambda(
    lambda text: text.upper()
)
```

Example:

```python
result = uppercase.invoke(
    "hello"
)

print(result)
```

Output:

```text
HELLO
```

But for learning and bigger projects, named functions are usually easier to read.

---

# 12. Why is it called `RunnableLambda`?

In LangChain, a **Runnable** is something that can be executed using methods like:

```text
invoke()
batch()
stream()
ainvoke()
```

A normal function:

```python
def clean_text(text):
    ...
```

is just a Python function.

After:

```python
RunnableLambda(clean_text)
```

it behaves like a LangChain runnable.

So:

```text
Python Function
      ↓
RunnableLambda
      ↓
LangChain Runnable
```

---

# 13. You Can Call `invoke()` Directly

Example:

```python
def double_number(number):
    return number * 2


double = RunnableLambda(
    double_number
)


result = double.invoke(10)


print(result)
```

Output:

```text
20
```

So `RunnableLambda` does not need to be inside a full chain.

---

# 14. `batch()` with `RunnableLambda`

Because it is now a Runnable, you can also use:

```python
results = double.batch([
    10,
    20,
    30
])


print(results)
```

Output:

```python
[
    20,
    40,
    60
]
```

---

# 15. Real RAG Use Case

One very common use of `RunnableLambda` in RAG is formatting retrieved documents.

Imagine the retriever returns:

```python
[
    Document(page_content="Chunk 1..."),
    Document(page_content="Chunk 2..."),
    Document(page_content="Chunk 3...")
]
```

The prompt does not need raw `Document` objects.

It needs text.

So we can create:

```python
def format_documents(documents):

    return "\n\n".join(
        doc.page_content
        for doc in documents
    )
```

Then:

```python
format_docs = RunnableLambda(
    format_documents
)
```

Flow:

```text
Retriever
    ↓
List of Documents
    ↓
RunnableLambda
    ↓
format_documents()
    ↓
One context string
    ↓
Prompt
```

This is a very important use case for your future RAG project.

---

# 16. Example RAG Formatting Code

```python
from langchain_core.runnables import RunnableLambda


def format_documents(documents):

    return "\n\n".join(
        doc.page_content
        for doc in documents
    )


format_docs = RunnableLambda(
    format_documents
)
```

Then later:

```python
retriever | format_docs
```

means:

```text
Question
   ↓
Retriever
   ↓
Relevant Documents
   ↓
RunnableLambda
   ↓
Formatted Context
```

---

# 17. `RunnableLambda` vs Normal Function

## Normal function

```python
def clean_text(text):
    return text.strip()
```

You call:

```python
clean_text(" hello ")
```

---

## RunnableLambda

```python
cleaner = RunnableLambda(
    clean_text
)
```

You call:

```python
cleaner.invoke(
    " hello "
)
```

And now it can also be used inside:

```python
prompt | model | parser | cleaner
```

---

# 18. When Should I Use `RunnableLambda`?

Use it when you need custom Python logic inside your LangChain pipeline.

Examples:

```text
Clean input
Format output
Format documents
Extract dictionary values
Convert data types
Calculate values
Add/remove fields
Transform retrieved context
Filter data
Pre-process input
Post-process output
```

---

# 19. When Do I Not Need It?

If you are only doing:

```text
Prompt
 ↓
Model
 ↓
Parser
```

you do not need `RunnableLambda`.

Example:

```python
chain = prompt | model | parser
```

is already enough.

Use `RunnableLambda` only when you need custom logic.

---

# 20. Main Mental Model

```text
Normal Python Function
        ↓
RunnableLambda
        ↓
Can be used in LCEL
        ↓
Can use invoke(), batch(), etc.
```

Example:

```python
def clean(text):
    return text.strip()


cleaner = RunnableLambda(clean)
```

Then:

```python
chain = (
    prompt
    | model
    | parser
    | cleaner
)
```

---

# 21. Quick Summary

## `RunnableLambda`

```text
Wraps a normal Python function
so it can be used as a LangChain Runnable.
```

## Basic syntax

```python
runnable = RunnableLambda(
    my_function
)
```

## Run once

```python
runnable.invoke(input)
```

## Use inside a chain

```python
A | RunnableLambda(function) | B
```

## Common RAG use

```text
Retriever
   ↓
Documents
   ↓
RunnableLambda
   ↓
Formatted Context
```

---

# 22. Most Important Thing to Remember

```python
def my_function(data):
    return something


my_runnable = RunnableLambda(
    my_function
)
```

means:

```text
Python function
     ↓
Convert to LangChain Runnable
     ↓
Use inside a chain
```

The input and output types must also make sense for the surrounding chain.

Example:

```text
StrOutputParser
      ↓
String
      ↓
RunnableLambda expecting String
```

is correct.

---

# 23. Final Example to Remember

```python
from langchain_core.runnables import RunnableLambda


def make_uppercase(text):
    return text.upper()


uppercase = RunnableLambda(
    make_uppercase
)


chain = (
    prompt
    | model
    | StrOutputParser()
    | uppercase
)
```

Flow:

```text
Prompt
 ↓
Model
 ↓
AIMessage
 ↓
StrOutputParser
 ↓
String
 ↓
RunnableLambda
 ↓
Python Function
 ↓
Final Output
```
