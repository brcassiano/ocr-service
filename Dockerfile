FROM python:3.11-slim

# Cache bust para forçar rebuild
ENV CACHE_BUST=2025-12-07-v3

# Metadados
LABEL maintainer="MEIre Team"
LABEL description="OCR Service para comprovantes fiscais brasileiros"

# Instalar dependências do sistema (CORRIGIDO para Debian Trixie)
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd -m -u 1000 appuser

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY ./app ./app

# Mudar ownership para appuser
RUN chown -R appuser:appuser /app

# Trocar para usuário não-root
USER appuser

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]