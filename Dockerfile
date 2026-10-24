FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    WEB_CONCURRENCY=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev \
    && useradd --create-home --uid 10001 --user-group app

COPY app/ ./app/
COPY train_model.py ./
RUN python train_model.py

USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
