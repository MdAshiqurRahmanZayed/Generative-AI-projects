# Generative AI Projects

Collection of generative AI experiments and small apps: Gemini + LangChain notebooks, a DeepSeek/Ollama Streamlit chatbot, a LangChain restaurant-name generator, and misc practice notebooks.

## Structure

- `Gemini+LangChain/` — notebooks using Gemini via `langchain-google-genai` (chat, image generation, Chroma-backed retrieval, a command runner).
- `DeepSeekWithStreamlit/` — Streamlit chat app backed by a local Ollama model (`langchain-ollama`).
- `Restaurant_Name_Generator/` — Streamlit app generating restaurant names/menu items via Gemini + LangChain chains.
- `Practice/` — scratch/learning notebooks.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+ (uv will install the interpreter if missing).

```bash
# install deps into a local .venv
uv sync

# activate (optional, uv run works without activating)
source .venv/bin/activate
```

### Environment variables

Copy `.env-exemple` to `.env` and fill in your key:

```bash
cp .env-exemple .env
```

```
GOOGLE_API_KEY=your-gemini-api-key
```

## Running things

Notebooks (Gemini+LangChain, Practice):

```bash
uv run jupyter lab
```

DeepSeek Streamlit app (needs [Ollama](https://ollama.com) running locally with the model pulled):

```bash
cd DeepSeekWithStreamlit
uv run streamlit run main.py
```

Restaurant Name Generator:

```bash
cd Restaurant_Name_Generator
uv run streamlit run main.py
```

## Adding dependencies

```bash
uv add <package>
```

This updates `pyproject.toml` and `uv.lock`. Commit both.
