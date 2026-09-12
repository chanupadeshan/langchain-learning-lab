# LangChain Learning Examples

A small collection of runnable LangChain examples and their accompanying study
notes.

## Project structure

```text
.
├── topics/
│   ├── prompt_model_parser/
│   │   ├── README.md
│   │   └── example.py
│   ├── memory/
│   │   ├── normal_memory/
│   │   │   ├── README.md
│   │   │   └── example.py
│   │   └── redis_memory/
│   │       ├── README.md
│   │       └── example.py
│   ├── runnable_lambda/
│   │   ├── README.md
│   │   └── example.py
│   └── structured_output/
│       ├── README.md
│       └── example.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

Every topic folder is self-contained: `README.md` explains the concept and
`example.py` provides runnable code.

The memory topic contains separate normal in-memory and persistent Redis
examples.

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:

```dotenv
GROQ_API_KEY=your_api_key_here
```

Run an example from the project root:

```bash
python topics/prompt_model_parser/example.py
```
