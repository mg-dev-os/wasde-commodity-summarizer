# WASDE Commodity Summarizer

Process USDA WASDE PDF reports and summarize commentary by commodity (Wheat, Coarse Grains, Rice, etc.) using an LLM.

Supports **OpenAI** (cloud, paid), **Groq** (free cloud), or **Google Gemini** (free tier).

## Setup

1. **Clone and install**

   ```bash
   cd assignment
   pip install -r requirements.txt
   ```

2. **Choose LLM provider** (in `config.yaml`)

   - **Groq (free cloud)** – default. Get a free API key at [console.groq.com/keys](https://console.groq.com/keys). Add to `.env`:
     ```
     GROQ_API_KEY=your-key
     ```
     In `config.yaml`:
     ```yaml
     llm:
       provider: "groq"
     ```

   - **Google Gemini (free tier)** – Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey). Add to `.env`:
     ```
     GEMINI_API_KEY=your-key
     ```
     In `config.yaml`:
     ```yaml
     llm:
       provider: "gemini"
     ```

   - **OpenAI** – when you have quota. Add `OPENAI_API_KEY` to `.env`. In `config.yaml` set `llm.provider: "openai"`. Do **not** commit `.env`.

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Open the URL shown (e.g. http://localhost:8501).

## Docker

You can run the app in a container so you don’t need to install Python or dependencies on the host.

**Using Docker Compose (recommended)**

1. Create a `.env` file in the project root with your API key, for example:
   ```
   GROQ_API_KEY=your-groq-key
   ```
2. From the project root:
   ```bash
   docker compose up --build
   ```
3. Open http://localhost:8501 in your browser.

**Using Docker only**

1. Build the image:
   ```bash
   docker build -t wasde-summarizer .
   ```
2. Run the container, passing your API key (or use `--env-file .env`):
   ```bash
   docker run -p 8501:8501 -e GROQ_API_KEY=your-key wasde-summarizer
   ```
3. Open http://localhost:8501.

The Dockerfile uses `python:3.11-slim`, installs dependencies from `requirements.txt`, and runs Streamlit on port 8501. Use `config.yaml` in the image to set `llm.provider` (groq / gemini / openai); API keys are provided via environment variables (e.g. from `.env` or `-e`).

## Usage

- **Upload**: Choose a WASDE PDF (e.g. from [USDA WASDE](https://www.usda.gov/oce/commodity/wasde)).
- **Extract**: The app extracts text and splits it by commodity section headers.
- **Summarize**: Click "Generate summaries" to send each section to the configured LLM and show results in the UI.

## Configuration

- **config.yaml**: `llm.provider` (`groq`, `gemini`, or `openai`), model names, token limits, commodity headers, UI settings.
- **Environment**: `GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` depending on provider.

## Tests

```bash
pytest tests/ -v
```

Tests use mocks for the OpenAI API (no real key required for tests).

## Project layout

- `app.py` – Streamlit UI
- `src/` – Config loader, PDF extractor, summarizer
- `config.yaml` – Settings (including `llm.provider`)
- `Dockerfile` – Image for running the app in Docker
- `docker-compose.yml` – Compose file (uses `.env` for API keys)
- `JUSTIFICATION.md` – Why OpenAI direct API vs LangChain/LiteLLM

## Deliverables

- **GitHub**: Push this repo and share the repository link.
- **Demo video**: Record a short screen capture: upload a WASDE PDF, run summarization, show results.
