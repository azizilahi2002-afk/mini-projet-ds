# ===== Stage 1 : Builder =====
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt requirements-dev.txt ./

RUN pip install  \
    -r requirements.txt \
    -r requirements-dev.txt


# ===== Stage 2 : Runtime =====
FROM python:3.11-slim

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models /app/data && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copier les dépendances installées globalement
COPY --from=builder /usr/local /usr/local

# Copier le code
COPY --chown=appuser:appuser src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
