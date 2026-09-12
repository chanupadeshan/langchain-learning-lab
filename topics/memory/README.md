# LangChain Conversation Memory

These examples demonstrate session-based conversation history with LangChain's
`RunnableWithMessageHistory`.

| Memory type | Storage | Survives restart? | Best use |
| --- | --- | --- | --- |
| [Normal memory](normal_memory/) | Application process | No | Learning and local testing |
| [Redis memory](redis_memory/) | Redis list | Yes | Durable, shared chat history |

Both examples use a `session_id` to identify a conversation. Calls using the
same `session_id` share history; a different ID starts a separate conversation.
