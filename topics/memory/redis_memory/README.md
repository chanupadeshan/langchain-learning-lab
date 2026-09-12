# Redis Conversation Memory

This tutorial explains how `example.py` stores LangChain chat history in Redis
using `RedisChatMessageHistory` and `RunnableWithMessageHistory`.

It follows the same prompt and conversation flow as the normal-memory example.
The important difference is where the messages are stored:

```text
Normal memory → Python dictionary → lost when the process stops
Redis memory  → Redis database    → available after process restarts
```

## What the example demonstrates

The program sends two messages:

1. `My name is Chanupa.`
2. `What is my name?`

Both calls use session ID `1234`. Redis stores the first user message and AI
response. The history wrapper loads those messages before the second call, so
the model can recall the name.

## Conversation flow

```text
Input dictionary
    ↓
RunnableWithMessageHistory reads session_id "1234"
    ↓
get_memory() creates a RedisChatMessageHistory view for that session
    ↓
Messages are loaded from Redis
    ↓
ChatPromptTemplate inserts them at MessagesPlaceholder
    ↓
ChatGroq generates a response
    ↓
StrOutputParser converts the response to text
    ↓
The new human and AI messages are saved in Redis
    ↓
Final string
```

## Requirements

This example needs:

- Python dependencies `langchain-core`, `langchain-groq`,
  `langchain-community`, `redis`, and `python-dotenv`
- A running Redis server reachable from the computer
- `GROQ_API_KEY` and `REDIS_URL` in the root `.env` file

This specific community adapter stores messages in a normal Redis list. It does
not require a Redis search index, RedisJSON, RediSearch, or
`checkpointer.setup()`.

Install the project dependencies from the root directory:

```bash
pip install -r requirements.txt
```

## Environment variables

The `.env` file should contain:

```dotenv
GROQ_API_KEY=your_api_key_here
REDIS_URL=redis://localhost:6379/0
```

Authenticated Redis URLs commonly look like:

```dotenv
REDIS_URL=redis://username:password@hostname:6379/0
```

Use `rediss://` instead of `redis://` when the Redis provider requires TLS:

```dotenv
REDIS_URL=rediss://username:password@hostname:6379/0
```

Do not commit `.env`, paste its secrets into source code, or print the complete
Redis URL because it may contain a password.

## Imports

```python
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
```

- `os` reads `REDIS_URL` from the environment.
- `load_dotenv` loads the root `.env` file.
- `ChatGroq` provides the chat model.
- `ChatPromptTemplate` builds the system, history, and human messages.
- `MessagesPlaceholder` determines where old messages are inserted.
- `StrOutputParser` returns the final model response as a string.
- `RunnableWithMessageHistory` manages loading and saving chat history.
- `RedisChatMessageHistory` stores that history in Redis.

The example imports `RedisChatMessageHistory` from `langchain_community`. This is
different from the `langchain_redis` package, whose implementation uses a search
index and has different constructor behavior.

## Load configuration

```python
load_dotenv(override=True)
```

`override=True` makes values from `.env` replace values already exported in the
shell.

The Redis URL is then read and validated:

```python
redis_url = os.getenv("REDIS_URL")
if not redis_url:
    raise ValueError("REDIS_URL is missing from the .env file")
```

Failing immediately is safer than silently connecting to an unintended default
Redis database.

## Create the model

```python
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
```

`temperature=0` requests more consistent output. Redis stores the messages; the
model does not maintain the conversation by itself.

## Create the prompt

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
```

The final prompt order is:

1. System instruction
2. Messages loaded from Redis
3. Current human input

The name `history` must match `history_messages_key="history"`. The name `input`
must match `input_messages_key="input"` and the input dictionary used with
`invoke()`.

## Build the base chain

```python
parser = StrOutputParser()
chain = prompt | model | parser
```

The base chain performs prompt formatting, model invocation, and output parsing.
It does not access Redis until it is wrapped with message history.

## Create Redis history for a session

```python
def get_memory(session_id):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url,
    )
```

The history factory receives the session ID from the runnable configuration.
`RedisChatMessageHistory` combines its default key prefix with the ID.

For session `1234`, the default Redis key is:

```text
message_store:1234
```

Messages are serialized and stored in a Redis list. Creating a new Python
`RedisChatMessageHistory` instance does not erase the list; it provides access to
the messages already stored under that key.

The constructor parameter is named `url`, not `redis_url`, for this community
adapter.

## Add Redis memory to the chain

```python
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_memory,
    input_messages_key="input",
    history_messages_key="history",
)
```

Before each call, the wrapper:

1. Reads `session_id` from the configuration.
2. Calls `get_memory(session_id)`.
3. Loads previous messages from Redis.
4. Inserts them into the prompt's `history` placeholder.
5. Runs the base chain.
6. Stores the new human and AI messages in Redis.

## Select the conversation

```python
config = {
    "configurable": {
        "session_id": "1234"
    }
}
```

The session ID is the link between a conversation and its Redis key. Reusing
`1234` continues the existing conversation—even after the Python program has
stopped and restarted. A different ID uses a different Redis key and therefore
a separate history.

Real applications should generate a stable conversation ID rather than using a
single hard-coded value for every user.

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

Because both calls use the same session ID, the second prompt contains the first
turn. The parser makes each `result` a normal Python string.

The exact wording varies, but output should resemble:

```text
AI: Nice to meet you, Chanupa!
AI: Your name is Chanupa.
```

## Run the example

From the project root:

```bash
python topics/memory/redis_memory/example.py
```

Or from this folder:

```bash
python example.py
```

Redis must already be running and reachable through `REDIS_URL`.

## Important rerun behavior

Redis history survives program restarts. Running the tutorial repeatedly with
session ID `1234` keeps appending the same two messages to the existing history.
This can produce duplicated conversation turns.

For a clean tutorial run, either use a new session ID:

```python
"session_id": "1234-new"
```

or clear the existing session intentionally:

```python
history = RedisChatMessageHistory(session_id="1234", url=redis_url)
history.clear()
```

`clear()` deletes the Redis key for that session. Do not call it automatically in
an application where the conversation must persist.

## Add automatic expiration

The community adapter supports a time-to-live value in seconds:

```python
def get_memory(session_id):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url,
        ttl=3600,
    )
```

With `ttl=3600`, a session expires one hour after its latest added message. TTL
helps prevent abandoned chat histories from remaining forever.

## Use a custom key prefix

```python
RedisChatMessageHistory(
    session_id=session_id,
    url=redis_url,
    key_prefix="my_app:chat:",
)
```

For session `1234`, that produces the key:

```text
my_app:chat:1234
```

A distinct prefix prevents collisions when multiple applications share one
Redis database.

## Inspect a session safely

You can retrieve its LangChain message objects:

```python
history = get_memory("1234")

for message in history.messages:
    print(type(message).__name__, message.content)
```

Avoid printing real conversation content when it may contain personal or
sensitive data.

## Things to remember

- Redis stores the chat messages; the model itself does not remember them.
- Reuse the same `session_id` to continue a conversation.
- Use different session IDs to isolate conversations.
- `MessagesPlaceholder("history")` and `history_messages_key="history"` must
  match.
- `{input}`, `input_messages_key="input"`, and `invoke({"input": ...})` must
  match.
- This adapter expects `url=redis_url`, not `redis_url=redis_url`.
- The default Redis key is `message_store:<session_id>`.
- Redis memory persists across Python restarts unless the key is deleted or a
  TTL expires it.
- Repeated runs with a fixed session ID append duplicate tutorial messages.
- The example stores the entire conversation with no trimming or summarization.
  Long histories increase latency, token usage, and model-context consumption.
- Protect Redis with authentication, network restrictions, and TLS when it is
  not strictly local.
- Chat messages may contain sensitive information. Apply appropriate retention
  and access-control policies.
- `RunnableWithMessageHistory` is a legacy-style wrapper in newer LangChain
  releases. New production systems should also evaluate LangGraph persistence.

## Common errors

### `REDIS_URL is missing from the .env file`

Add `REDIS_URL` to the root `.env` file. Check the spelling and do not include
spaces before the variable name.

### Connection refused or timeout

Verify that Redis is running, the hostname and port are correct, the selected
database exists, firewall rules allow access, and the URL uses the required TLS
scheme.

### Authentication error

Check the Redis username and password embedded in `REDIS_URL`. URL-encode special
characters in credentials.

### The model forgets previous messages

Confirm that both invocations use the same `session_id`. Also verify that the
application is connecting to the same Redis server and database on each call.

### Old or unexpected messages appear

The fixed session ID already has persistent history. Use a new session ID or
explicitly clear that session.

### `dict` has no attribute `index`

That error comes from a different adapter imported from `langchain_redis`, where
some package combinations inspect a Redis search index incompatibly. This
example intentionally imports the list-based community adapter:

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
```

Keep the import and its `url=...` constructor form consistent with this tutorial.
