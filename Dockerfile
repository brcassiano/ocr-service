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

# Pré-baixar modelos do PaddleOCR durante build (evita corrupted cache)
RUN python -c "from paddleocr import PaddleOCR; OCREngine = PaddleOCR(use_angle_cls=False, lang='pt', show_log=False); print('Modelos baixados com sucesso')" || echo "Falha no download, será baixado no startup"

# Copiar código
COPY ./app ./app

EXPOSE 8000

# Startup: LIMPAR TODO CACHE + iniciar
CMD sh -c "\
    echo 'Limpando cache PaddleOCR...' && \
    rm -rf /root/.paddleocr 2>/dev/null || true && \
    rm -rf /home/appuser/.paddleocr 2>/dev/null || true && \
    rm -rf ~/.paddleocr 2>/dev/null || true && \
    echo 'Cache limpo! Iniciando servidor...' && \
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info"