FROM python:3.12-slim

WORKDIR /app

# build-essential + git: needed to build native wheels and pip-install eai-delphi from git
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, separately, so the heavy layer is cached independently
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY web ./web

ENV PORT=8000

# bootstrap is idempotent: first boot fetches features onto the volume, subsequent boots no-op
CMD ["sh", "-c", "python -m server.bootstrap && exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --workers 2"]
