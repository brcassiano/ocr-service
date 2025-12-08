FROM python:3.11-slim

LABEL maintainer="MEIre Team"

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pré-baixar modelos do PaddleOCR durante build
RUN python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=False, lang='pt', show_log=False); print('✓ Modelos PaddleOCR baixados')" || echo "Falha no download"

# Copiar código
COPY ./app ./app

# Garantir que não há bytecode cache
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -delete

EXPOSE 8000

# Startup com limpeza total
CMD sh -c "\
    echo '=== LIMPANDO CACHES ===' && \
    rm -rf /root/.paddleocr /home/appuser/.paddleocr ~/.paddleocr 2>/dev/null || true && \
    find /app -type f -name '*.pyc' -delete 2>/dev/null || true && \
    find /app -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    echo '=== INICIANDO SERVIDOR ===' && \
    python -u -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info"