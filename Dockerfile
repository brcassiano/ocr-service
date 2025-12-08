FROM python:3.11-slim

LABEL maintainer="MEIre Team"
LABEL description="OCR Service para comprovantes fiscais brasileiros"

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY ./app ./app

EXPOSE 8000

# Startup: limpar cache corrompido + iniciar servidor
CMD sh -c "rm -rf /root/.paddleocr/whl/cls /home/appuser/.paddleocr/whl/cls 2>/dev/null || true && uvicorn app.main:app --host 0.0.0.0 --port 8000"