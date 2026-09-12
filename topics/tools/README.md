# LangChain Basic Agent — Study Note

This note explains a simple LangChain agent using Groq and custom tools.

It also explains:

- what `@tool` does
- what `create_agent()` does
- how tool calling works
- what the agent loop is
- why we usually do **not** use `StrOutputParser`
- why we usually do **not** build the top-level agent with an LCEL chain such as `prompt | model | parser`

---

## 1. Full Agent Code

```python
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv


load_dotenv(override=True)


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


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


agent = create_agent(
    model=model,
    tools=[
        add_numbers,
        multiply_numbers,
        subtract_numbers
    ],
    system_prompt="""
    You are a helpful AI assistant.

    Use the available tools when calculation is needed.
    """
)


result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Multiply 20 by 5, then add 10."
        }
    ]
})


print(
    "Agent's response:",
    result["messages"][-1].content
)
```

---

## 2. What is an Agent?

A simple way to think about an agent is:

```text
Agent
=
LLM
+
Tools
+
Decision-making loop
```

The important difference from a normal chain is:

```text
Chain
→ YOU define the exact steps

Agent
→ the LLM decides which steps/tools are needed
```

---

## 3. Normal Chain vs Agent

### Normal chain

Example:

```python
chain = prompt | model | parser
```

The order is fixed:

```text
Input
 ↓
Prompt
 ↓
Model
 ↓
Parser
 ↓
Output
```

The same steps happen every time.

### Agent

An agent is dynamic:

```text
User
 ↓
Model
 ↓
Need tool?
 ├── No → Final Answer
 └── Yes
      ↓
     Tool
      ↓
 Tool Result
      ↓
    Model
      ↓
Need another tool?
```

The model decides what happens next.

---

## 4. What is `@tool`?

```python
from langchain_core.tools import tool
```

Example:

```python
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

`@tool` converts a normal Python function into a LangChain tool.

```text
Python Function
      ↓
    @tool
      ↓
LangChain Tool
```

The agent can now decide to call it.

---

## 5. Why is the Docstring Important?

Example:

```python
"""Add two numbers together."""
```

The model uses the tool's:

```text
name
description
parameters
```

to understand when it should use that tool.

For example:

```text
add_numbers
→ Add two numbers together

multiply_numbers
→ Multiply two numbers together

subtract_numbers
→ Subtract two numbers
```

The descriptions help the LLM choose the correct tool.

---

## 6. Tool Input Types

Example:

```python
def add_numbers(a: int, b: int) -> int:
```

This tells LangChain and the LLM:

```text
a → integer
b → integer
return value → integer
```

The type hints help generate the tool schema.

---

## 7. What is `create_agent()`?

```python
from langchain.agents import create_agent
```

Example:

```python
agent = create_agent(
    model=model,
    tools=[
        add_numbers,
        multiply_numbers,
        subtract_numbers
    ],
    system_prompt="..."
)
```

`create_agent()` combines:

```text
Model
+
Tools
+
Agent execution loop
```

Conceptually:

```text
ChatGroq
   +
Tools
   +
Decision Loop
   ↓
Agent
```

---

## 8. What is `system_prompt`?

```python
system_prompt="""
You are a helpful AI assistant.

Use the available tools when calculation is needed.
"""
```

This tells the agent how it should behave.

Think:

```text
system_prompt
=
agent instructions
```

---

## 9. Agent Input

We call the agent like this:

```python
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Multiply 20 by 5, then add 10."
        }
    ]
})
```

Conceptually:

```text
HumanMessage:
Multiply 20 by 5, then add 10.
```

---

## 10. What Happens Internally?

For:

```text
Multiply 20 by 5, then add 10.
```

the agent can do:

```text
User
 ↓
Model
 ↓
"I need multiplication"
 ↓
multiply_numbers(20, 5)
 ↓
100
 ↓
Model
 ↓
"I still need addition"
 ↓
add_numbers(100, 10)
 ↓
110
 ↓
Model
 ↓
Final Answer
```

That is the **agent loop**.

---

## 11. Agent Loop

```text
Model
 ↓
Tool Call
 ↓
Tool Result
 ↓
Model
 ↓
Tool Call
 ↓
Tool Result
 ↓
Model
 ↓
Final Answer
```

The loop continues until the model decides it does not need another tool.

---

## 12. Messages Produced by an Agent

The result may contain:

```text
HumanMessage
AIMessage
ToolMessage
AIMessage
ToolMessage
AIMessage
```

For our example:

```text
HumanMessage
"Multiply 20 by 5, then add 10."

AIMessage
Tool call:
multiply_numbers

ToolMessage
100

AIMessage
Tool call:
add_numbers

ToolMessage
110

AIMessage
"The result is 110."
```

---

## 13. Why Does `print(result)` Look Complicated?

If you do:

```python
print(result)
```

you print the entire agent state.

That contains:

```text
user message
tool calls
tool results
final AI response
metadata
token usage
```

That is why the output looks large.

---

## 14. Print Only the Final Answer

Use:

```python
result["messages"][-1].content
```

Explanation:

```python
result["messages"]
```

means all messages.

```python
[-1]
```

means the last message.

```python
.content
```

means the text inside the last `AIMessage`.

---

## 15. Why `.content` and Not `["content"]`?

The last item is an `AIMessage` object.

Correct:

```python
result["messages"][-1].content
```

Wrong:

```python
result["messages"][-1]["content"]
```

because:

```text
AIMessage ≠ dictionary
```

Think:

```text
Dictionary
→ ["content"]

AIMessage object
→ .content
```

---

## 16. Why Don't We Use `StrOutputParser()`?

In a normal chain:

```python
chain = prompt | model | StrOutputParser()
```

the model returns an `AIMessage`.

`StrOutputParser()` converts:

```text
AIMessage
 ↓
String
```

That is useful because the chain has one simple model output.

### Agent output is different

An agent returns something like:

```python
{
    "messages": [
        HumanMessage(...),
        AIMessage(tool_calls=...),
        ToolMessage(...),
        AIMessage(...)
    ]
}
```

So the top-level result is not just one `AIMessage`.

It is an **agent state containing many messages**.

Therefore, this is usually not what we want:

```python
agent | StrOutputParser()
```

Instead, we take the final message:

```python
result["messages"][-1].content
```

---

## 17. Parser Comparison

### Normal Chain

```python
chain = (
    prompt
    | model
    | StrOutputParser()
)
```

Result:

```text
String
```

### Agent

```python
result = agent.invoke(...)
```

Result:

```text
Agent State
└── messages
```

Final text:

```python
result["messages"][-1].content
```

So:

```text
Normal Chain
→ parser is often useful

Agent
→ usually access final AIMessage directly
```

---

## 18. Does an Agent Never Use a Parser?

Not exactly.

A parser can still be useful **inside a tool or sub-chain**.

Example:

```python
summary_chain = (
    prompt
    | model
    | StrOutputParser()
)
```

Then:

```python
@tool
def summarize_text(text: str) -> str:
    """Summarize text."""

    return summary_chain.invoke({
        "input": text
    })
```

Now:

```text
Agent
 ↓
Tool
 ↓
Chain
 ↓
Prompt
 ↓
Model
 ↓
Parser
```

So parsers are not forbidden.

They are just usually **not needed as the final top-level agent parser**.

---

## 19. Why Don't We Create a Normal LCEL Chain for the Agent?

A normal LCEL chain has a fixed order:

```python
chain = prompt | model | parser
```

That means:

```text
Prompt
 ↓
Model
 ↓
Parser
```

every time.

But an agent needs a loop:

```text
Model
 ↓
Tool
 ↓
Model
 ↓
Tool
 ↓
Model
```

The number and order of tool calls are not known in advance.

Therefore, a normal linear chain is not the right top-level structure for an agent.

---

## 20. Fixed Workflow vs Dynamic Workflow

### Chain

You decide:

```text
Step 1
 ↓
Step 2
 ↓
Step 3
```

Example:

```text
Prompt
 ↓
Model
 ↓
Parser
```

### Agent

The model decides:

```text
Should I call a tool?
Which tool?
What arguments?
Do I need another tool?
Can I answer now?
```

So:

```text
Chain = fixed workflow
Agent = dynamic workflow
```

---

## 21. Why Doesn't This Agent Need `ChatPromptTemplate`?

In a simple agent, you can use:

```python
system_prompt="..."
```

and pass the user input through:

```python
{
    "messages": [...]
}
```

So you do not need to manually build:

```python
ChatPromptTemplate
```

for this basic case.

A `ChatPromptTemplate` can still be useful inside tools or other chains.

---

## 22. Why Doesn't This Agent Need `StrOutputParser`?

Because the final response is already available as:

```python
result["messages"][-1].content
```

So the simple agent does not need:

```python
StrOutputParser()
```

at the top level.

---

## 23. Clean Agent Code

```python
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv


load_dotenv(override=True)


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


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


agent = create_agent(
    model=model,
    tools=[
        add_numbers,
        multiply_numbers,
        subtract_numbers
    ],
    system_prompt="""
    You are a helpful AI assistant.
    Use the available tools when calculation is needed.
    """
)


result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Multiply 20 by 5, then add 10."
        }
    ]
})


print(
    "Agent's response:",
    result["messages"][-1].content
)
```

---

## 24. Agent Architecture

```text
                  USER
                    │
                    ▼
                  AGENT
                    │
                    ▼
                  MODEL
                    │
              Need a tool?
             /           \
           No             Yes
           │               │
           ▼               ▼
     Final Answer       Choose Tool
                           │
                           ▼
                      Execute Tool
                           │
                           ▼
                      Tool Result
                           │
                           └───────┐
                                   │
                                   ▼
                                  MODEL
```

---

## 25. Agent vs Chain Summary

### Chain

```python
chain = prompt | model | parser
```

Use when:

```text
You know exactly what steps should happen.
```

### Agent

```python
agent = create_agent(
    model=model,
    tools=[...]
)
```

Use when:

```text
The LLM should decide which tools/actions are needed.
```

---

## 26. Parser vs Agent Output Summary

### Chain

```text
AIMessage
 ↓
StrOutputParser
 ↓
String
```

### Agent

```text
Agent State
 ↓
messages[-1]
 ↓
AIMessage
 ↓
.content
 ↓
String
```

---

## 27. Structured Agent Output

If you want predictable fields from an agent, use structured response support rather than `StrOutputParser()`.

Conceptually:

```text
Agent
 ↓
Structured Response Schema
 ↓
Predictable Object
```

Example schema:

```python
from pydantic import BaseModel


class FinalAnswer(BaseModel):
    answer: int
    explanation: str
```

This is different from `StrOutputParser()`, which only converts an `AIMessage` to a string.

---

## 28. Most Important Concepts

```text
@tool
→ Converts Python function into a LangChain tool

create_agent()
→ Combines model + tools + decision loop

Tool Calling
→ Model decides which tool to call

ToolMessage
→ Result returned by a tool

Agent Loop
→ Model → Tool → Model → Tool → Final Answer

result["messages"][-1].content
→ Gets the final agent response
```

---

## 29. Most Important Difference to Remember

```text
CHAIN
You control the workflow.

AGENT
The LLM controls the workflow.
```

And:

```text
CHAIN
prompt | model | parser

AGENT
model + tools + decision loop
```

---

## 30. Final Mental Model

### Normal LangChain Chain

```text
Input
 ↓
Prompt
 ↓
Model
 ↓
Parser
 ↓
Output
```

### LangChain Agent

```text
User
 ↓
Model
 ↓
Choose Tool?
 ↓
Tool
 ↓
Tool Result
 ↓
Model
 ↓
Choose Another Tool?
 ↓
...
 ↓
Final AIMessage
 ↓
.content
```
