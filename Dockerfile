# WASDE Commodity Summarizer – run with Docker
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and config
COPY app.py config.yaml ./
COPY src ./src

# Streamlit listens on all interfaces so the host can reach it
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

# API keys and llm.provider come from env (e.g. docker run -e GROQ_API_KEY=... or docker-compose env_file)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
