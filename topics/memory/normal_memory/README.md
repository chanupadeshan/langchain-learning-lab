# Normal In-Memory Conversation Memory

This tutorial explains how `example.py` gives a LangChain conversation short-term
memory using `InMemoryChatMessageHistory` and `RunnableWithMessageHistory`.

## What the example demonstrates

The program has two conversations turns:

1. The user says, `My name is Chanupa.`
2. The user asks, `What is my name?`

Both calls use session ID `1234`. The first user message and the first AI response
are saved in memory. Before the second call, LangChain loads those messages into
the prompt, allowing the model to answer the question using the earlier turn.

The memory exists only inside the running Python process. It is lost when the
program stops.

## Conversation flow

```text
Input dictionary
    ↓
RunnableWithMessageHistory finds session "1234"
    ↓
get_memory() returns that session's history
    ↓
ChatPromptTemplate inserts history at MessagesPlaceholder
    ↓
ChatGroq generates an AIMessage
    ↓
StrOutputParser converts the AIMessage to a string
    ↓
RunnableWithMessageHistory saves the user and AI messages
    ↓
Final string
```

## Imports

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
```

- `load_dotenv` loads environment variables from `.env`.
- `ChatGroq` connects LangChain to the Groq chat-model API.
- `ChatPromptTemplate` creates a reusable chat prompt.
- `MessagesPlaceholder` marks the exact location where past messages are added.
- `StrOutputParser` converts the model's `AIMessage` into plain text.
- `InMemoryChatMessageHistory` stores messages in the Python process.
- `RunnableWithMessageHistory` loads and saves history around a runnable chain.

## Load the API key

```python
load_dotenv(override=True)
```

The root `.env` file must contain:

```dotenv
GROQ_API_KEY=your_api_key_here
```

`override=True` means the value in `.env` replaces an existing
`GROQ_API_KEY` from the shell environment.

Never commit `.env` or print the API key.

## Create the model

```python
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
```

- `model` selects the Groq-hosted model.
- `temperature=0` requests more consistent, less random answers.
- Memory is not stored inside the model. The application stores messages and
  sends them back to the model on later calls.

## Create the prompt

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
```

The prompt is assembled in this order:

1. The system instruction
2. Previous human and AI messages from `history`
3. The new human message from `input`

The placeholder name `history` must match `history_messages_key="history"`
when the history wrapper is created.

The template variable `{input}` must match `input_messages_key="input"` and the
key supplied to `invoke()`.

## Build the base chain

```python
parser = StrOutputParser()
chain = prompt | model | parser
```

The `|` operator is LangChain Expression Language syntax. Data moves through:

```text
prompt → model → parser
```

The base chain does not manage memory by itself. Memory is added by wrapping it.

## Create the in-memory store

```python
memory = {}
```

This dictionary maps each session ID to a separate history object. Conceptually:

```python
{
    "1234": InMemoryChatMessageHistory(...),
    "5678": InMemoryChatMessageHistory(...),
}
```

Then the history factory creates or retrieves the correct object:

```python
def get_memory(session_id):
    if session_id not in memory:
        memory[session_id] = InMemoryChatMessageHistory()
    return memory[session_id]
```

Returning the same object for the same session is essential. Creating a new
history object on every call would make the second request forget the first.

## Add memory to the chain

```python
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_memory,
    input_messages_key="input",
    history_messages_key="history",
)
```

The arguments mean:

- `chain`: the runnable that should receive conversation history.
- `get_memory`: the function used to locate history for a session.
- `input_messages_key="input"`: the dictionary key containing the new message.
- `history_messages_key="history"`: the prompt key receiving old messages.

`RunnableWithMessageHistory` automatically reads the history before invocation
and saves the new human and AI messages after invocation.

## Select a conversation

```python
config = {
    "configurable": {
        "session_id": "1234"
    }
}
```

`session_id` identifies one conversation. Reusing `1234` continues that
conversation. Changing it starts or continues a different conversation.

In a real application, the session ID could be a generated conversation ID. Do
not use a password, API key, or other secret as a session ID.

## Invoke the conversation

First turn:

```python
result = chain_with_memory.invoke(
    {"input": "My name is Chanupa."},
    config,
)
```

Second turn:

```python
result = chain_with_memory.invoke(
    {"input": "What is my name?"},
    config,
)
```

The second call uses the same configuration, so its prompt includes the first
turn. Because the output parser returns text, `result` is a string.

The exact wording varies, but output should resemble:

```text
AI: Nice to meet you, Chanupa!
AI: Your name is Chanupa.
```

## Run the example

From the project root:

```bash
python topics/memory/normal_memory/example.py
```

Or from this folder:

```bash
python example.py
```

## Start another session

Use a different ID:

```python
other_config = {
    "configurable": {
        "session_id": "5678"
    }
}
```

Session `5678` cannot see messages stored under session `1234`.

## Clear a session

```python
memory["1234"].clear()
```

To remove the history object completely:

```python
memory.pop("1234", None)
```

## Things to remember

- The model itself does not remember. Earlier messages are stored and inserted
  into later prompts.
- `MessagesPlaceholder("history")` and `history_messages_key="history"` must
  use the same name.
- `{input}`, `input_messages_key="input"`, and `invoke({"input": ...})` must
  use the same name.
- Calls must reuse the same `session_id` to share history.
- Each session ID receives isolated history.
- This memory disappears when the process exits or restarts.
- Separate worker processes do not share the Python dictionary.
- History grows with every turn. Long conversations consume more model context
  and tokens, so production applications should trim or summarize history.
- An in-memory dictionary is appropriate for tutorials and local prototypes,
  not durable production storage.
- `RunnableWithMessageHistory` is a legacy-style history wrapper in newer
  LangChain releases. For new production systems, also study LangGraph
  checkpointers and short-term memory.

## Common mistakes

### The model forgets the first message

Check that both calls use the same `session_id` and that `get_memory()` returns
the existing object instead of creating a new one every time.

### A prompt variable is missing

Verify that the three names line up:

```text
input   ↔ input_messages_key ↔ invoke input
history ↔ history_messages_key ↔ MessagesPlaceholder
```

### The API key is missing

Confirm that `.env` is at the project root and contains `GROQ_API_KEY`.

### Memory usage keeps increasing

The example has no message limit. Clear inactive sessions, add expiration at the
application level, or use a strategy that trims or summarizes older messages.
