FROM python:3.11-slim

# Desabilitar bytecode cache do Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

LABEL maintainer="MEIre Team"

# 1. Instalar dependências do sistema
# Adicionamos git, gnupg e wget que o Playwright pode precisar, além das libs do OCR
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. INSTALAR PLAYWRIGHT + CHROMIUM (Passo novo)
# Isso baixa o navegador Chromium e instala as dependências de sistema dele
RUN playwright install --with-deps chromium

# 4. Pré-baixar modelos PaddleOCR (Mantido)
RUN python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=False, lang='pt', show_log=False); print('✓ Modelos baixados')" || echo "Falha no download"

# 5. Copiar código
COPY ./app ./app

# Remover qualquer .pyc existente
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -delete

EXPOSE 8000

# 6. Startup (Mantido, apenas executa o uvicorn)
CMD sh -c "\
    echo '=== LIMPANDO CACHES ===' && \
    rm -rf /root/.paddleocr /home/appuser/.paddleocr ~/.paddleocr 2>/dev/null || true && \
    find /app -type f -name '*.pyc' -delete 2>/dev/null || true && \
    find /app -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    echo '=== INICIANDO SERVIDOR (SEM BYTECODE CACHE) ===' && \
    exec python -B -u -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info --reload"