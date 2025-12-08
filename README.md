# OCR Service

Serviço de OCR especializado em comprovantes fiscais brasileiros, desenvolvido para integração com n8n.

## Características

- ✅ Extração de QR Codes (NFC-e)
- ✅ OCR de alta precisão (PaddleOCR)
- ✅ Detecção inteligente de gastos vs vendas
- ✅ Suporte a cupons fiscais, recibos manuscritos e comprovantes PIX
- ✅ API REST com FastAPI
- ✅ Pronto para Docker

## Endpoints

### POST /api/ocr/comprovante
Extrai dados estruturados de comprovantes.

**Request:**
curl -X POST http://localhost:8000/api/ocr/comprovante
-F "file=@cupom.jpg"

text

**Response:**
{
"tipo_documento": "gasto",
"itens": [
{
"item": "Pão Francês",
"quantidade": 1,
"valor_unitario": 8.50,
"valor_total": 8.50,
"data_compra": "07/12/2025"
}
],
"qrcode_url": "https://...",
"confianca": 0.956
}

text

### POST /api/ocr/qrcode-only
Extrai apenas QR Code (mais rápido).

### GET /health
Health check.