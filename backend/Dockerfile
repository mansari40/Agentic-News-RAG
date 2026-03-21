FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy everything needed for install
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY backend/ ./backend/

# Install Python dependencies
RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
