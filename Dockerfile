FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY . .

RUN mkdir -p models

EXPOSE 8082

CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"]
