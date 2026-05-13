FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install delphi without its declared deps: HEAD/0.1.3 pulls vllm + nvidia-*
# (~3 GB) that we don't use at runtime. requirements.txt covers what we need.
# delphi.clients/__init__.py eagerly imports Offline -> vllm; drop that line so
# `from delphi.clients import OpenRouter` works without vllm installed.
RUN pip install --no-cache-dir --no-deps eai-delphi==0.1.3 \
 && SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
 && printf "from .client import Client\nfrom .openrouter import OpenRouter\n__all__ = ['Client', 'OpenRouter']\n" \
      > "$SITE/delphi/clients/__init__.py" \
 && : > "$SITE/delphi/scorers/__init__.py" \
 && : > "$SITE/delphi/scorers/classifier/__init__.py" \
 && : > "$SITE/delphi/scorers/classifier/prompts/__init__.py"

COPY server ./server
COPY web ./web

ENV PORT=8000

CMD ["sh", "-c", "python -m server.bootstrap && exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --workers 2"]
