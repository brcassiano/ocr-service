from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from .ocr_engine import OCREngine
from .models import OCRResponse, QRCodeResponse, HealthResponse
from . import __version__
from .nfce_parser import NfceParserSP

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="MEIre OCR Service",
    description="Serviço de OCR especializado em comprovantes fiscais brasileiros",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recursos globais
ocr_engine: OCREngine | None = None
nfce_parser = NfceParserSP()

@app.on_event("startup")
async def startup_event():
    """Inicializa recursos na startup"""
    global ocr_engine
    logger.info("Iniciando MEIre OCR Service...")
    try:
        ocr_engine = OCREngine(use_gpu=False)
        logger.info("OCR Engine inicializado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao inicializar OCR Engine: {e}")
        raise

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "service": "meire-ocr-service",
        "version": __version__,
    }

# ----------------------------------------------------------------------
# ENDPOINT ANTIGO – OCR COMPLETO (mantido como fallback)
# ----------------------------------------------------------------------
@app.post("/api/ocr/comprovante", response_model=OCRResponse)
async def extract_comprovante(file: UploadFile = File(...)):
    """
    Extrai dados estruturados de comprovantes (OCR direto).
    Mantido como fallback para casos sem QR Code ou outros formatos.
    """
    try:
        logger.info(f"Processando arquivo: {file.filename} ({file.content_type})")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Arquivo deve ser uma imagem (JPEG, PNG, etc)",
            )

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Arquivo vazio")

        logger.info(f"Imagem carregada: {len(image_bytes)} bytes")

        qr_data = ocr_engine.extract_qrcode(image_bytes)
        ocr_result = ocr_engine.extract_text(image_bytes)
        structured_data = ocr_engine.structure_data(ocr_result, qr_data)

        logger.info(
            "Processamento concluído: tipo=%s, itens=%d",
            structured_data["tipo_documento"],
            len(structured_data["itens"]),
        )

        return JSONResponse(content=structured_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao processar comprovante: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": None,
                "mensagem": f"Erro interno ao processar imagem: {str(e)}",
                "confianca": 0.0,
            },
            status_code=200,
        )

# ----------------------------------------------------------------------
# ENDPOINT ANTIGO – SOMENTE QR CODE
# ----------------------------------------------------------------------
@app.post("/api/ocr/qrcode-only", response_model=QRCodeResponse)
async def extract_qrcode_only(file: UploadFile = File(...)):
    """Extrai apenas QR Code da imagem."""
    try:
        logger.info(f"Extraindo QR Code de: {file.filename}")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Arquivo deve ser uma imagem",
            )

        image_bytes = await file.read()
        qr_data = ocr_engine.extract_qrcode(image_bytes)

        if not qr_data:
            return {"found": False, "data": None}

        logger.info("QR Code extraído com sucesso")
        return {"found": True, "data": qr_data[0]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao extrair QR Code: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------------------------
# NOVO – IMAGEM → QR → NFC-e (SP)
# ----------------------------------------------------------------------
@app.post("/api/nfce/from-image", response_model=OCRResponse)
async def nfce_from_image(file: UploadFile = File(...)):
    """
    Fluxo recomendado para NFC-e:
    - lê imagem
    - extrai QR Code
    - consulta NFC-e na SEFAZ/SP
    - devolve itens diretamente da nota (sem OCR dos itens).
    """
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Arquivo vazio")

        # 1) QR Code
        qr_data = ocr_engine.extract_qrcode(image_bytes)
        if not qr_data:
            return JSONResponse(
                content={
                    "tipo_documento": "erro",
                    "itens": [],
                    "qrcode_url": None,
                    "mensagem": "Não consegui ler o QR Code da imagem.",
                    "confianca": 0.0,
                },
                status_code=200,
            )

        qr_url = qr_data[0]["data"]
        logger.info(f"QR Code URL extraída: {qr_url[:120]}...")

        # 2) NFC-e via web scraping / parser [web:106][web:109]
        nfce_data = await nfce_parser.parse(qr_url)

        response = {
            "tipo_documento": nfce_data.get("tipo_documento", "gasto"),
            "itens": nfce_data.get("itens", []),
            "qrcode_url": qr_url,
            "mensagem": None,
            "confianca": 1.0,  # dados oficiais da nota
        }
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao processar NFC-e via imagem: {e}", exc_info=True)
        return JSONResponse(
            content={
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": None,
                "mensagem": f"Erro interno ao consultar NFC-e: {str(e)}",
                "confianca": 0.0,
            },
            status_code=200,
        )

# ----------------------------------------------------------------------
# NOVO – URL NFC-e direta (opcional)
# ----------------------------------------------------------------------
class NfceUrlRequest(BaseModel):
    url: str

@app.post("/api/nfce/from-url", response_model=OCRResponse)
async def nfce_from_url(body: NfceUrlRequest):
    """Recebe a URL pública da NFC-e (do QR Code) e retorna itens estruturados."""
    try:
        nfce_data = await nfce_parser.parse(body.url)
        response = {
            "tipo_documento": nfce_data.get("tipo_documento", "gasto"),
            "itens": nfce_data.get("itens", []),
            "qrcode_url": body.url,
            "mensagem": None,
            "confianca": 1.0,
        }
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Erro ao processar NFC-e via URL: {e}", exc_info=True)
        return JSONResponse(
            content={
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": body.url,
                "mensagem": f"Erro interno ao consultar NFC-e: {str(e)}",
                "confianca": 0.0,
            },
            status_code=200,
        )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=2,
    )