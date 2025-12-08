FROM python:3.11-slim

# Cache bust
ENV CACHE_BUST=2025-12-07-v5

LABEL maintainer="MEIre Team"

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

# Script de inicialização que limpa cache corrompido
RUN echo '#!/bin/bash\n\
# Remover cache corrompido do PaddleOCR\n\
rm -rf /home/appuser/.paddleocr 2>/dev/null || true\n\
rm -rf ~/.paddleocr 2>/dev/null || true\n\
\n\
# Iniciar aplicação\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8000

# Usar script de inicialização
CMD ["/app/start.sh"]
