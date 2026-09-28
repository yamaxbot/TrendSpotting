FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/home/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vsellm_chat.py ./
COPY on_demand_parsing/ ./on_demand_parsing/
COPY preprocessing/create_features.py ./preprocessing/create_features.py
COPY website/ ./website/
COPY outputs/catboost_splits/random/catboost_model.cbm ./outputs/catboost_splits/random/catboost_model.cbm
COPY outputs/catboost_splits/random/metrics.json ./outputs/catboost_splits/random/metrics.json

RUN python website/manage.py collectstatic --noinput \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /home/app --create-home app \
    && mkdir -p /app/website/data /app/website/logics "$HF_HOME" \
    && chown -R app:app /app/website/data /app/website/logics /home/app

USER app
WORKDIR /app/website
EXPOSE 8000

CMD ["gunicorn", "trendspotting.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
