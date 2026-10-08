FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -U pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && pip install -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY dashboard_app/ dashboard_app/
RUN pip install --no-deps --no-build-isolation .

EXPOSE 7860
ENV PORT=7860
ENV HF_REPO_ID=mahdisabetkish/tcga-pancancer-set-transformer

CMD ["python3", "dashboard_app/app.py"]
