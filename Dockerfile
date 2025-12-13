FROM python:3.11-slim

# Desabilitar bytecode cache do Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

LABEL maintainer="MEIre Team"

# 1. Instalar dependências do sistema
# Apenas o necessário para PaddleOCR e ZBar (QR Code)
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Pré-baixar modelos PaddleOCR (Cache no build)
# Isso evita download na hora do startup
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='pt', show_log=False); print('✓ Modelos baixados')" || echo "Falha no download"

# 4. Copiar código da aplicação
COPY ./app ./app

# Remover lixo de cache Python
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -delete

EXPOSE 8000

# 5. Startup Otimizado
CMD sh -c "\
    echo '=== INICIANDO MEIRE OCR SERVICE ===' && \
    exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info"