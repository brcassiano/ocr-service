import logging
import uvicorn

from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .ocr_engine import OCREngine
from .nfce_parser import NfceParserSP
from .models import QRCodeResponse, HealthResponse
from . import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MEIre OCR Service",
    description="Serviço de OCR especializado em comprovantes fiscais brasileiros",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_engine: OCREngine | None = None


@app.on_event("startup")
async def startup_event():
    global ocr_engine
    logger.info("Iniciando MEIre OCR Service...")
    ocr_engine = OCREngine(use_gpu=False)
    logger.info("OCR Engine inicializado com sucesso")


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "service": "meire-ocr-service",
        "version": __version__,
    }


class QrCodeBody(BaseModel):
    qrcode_url: str


@app.post("/api/nfce/from-qrcode")
async def nfce_from_qrcode(body: QrCodeBody):
    # Debug ligado para inspecionar "text_head/flags" quando vier itens=[]
    # (pode desligar depois, passando enable_debug=False)
    parser = NfceParserSP(enable_debug=True)

    try:
        data = await parser.parse(body.qrcode_url)

        # normaliza payload
        data["qrcode_url"] = body.qrcode_url
        data["confianca"] = 1.0 if data.get("itens") else 0.0
        data.setdefault("tipo_documento", "gasto")
        data.setdefault("itens", [])
        data.setdefault("total_nota", None)
        data.setdefault("data_compra", None)
        data.setdefault("origem", "nfce_sp_qrcode")

        # garante debug sempre presente (evita “sempre a mesma resposta” sem pista)
        data.setdefault(
            "debug",
            {
                "enabled": True,
                "note": "Parser não retornou bloco de debug (verifique se o nfce_parser.py em runtime é o correto).",
            },
        )

        return JSONResponse(content=data)

    except Exception as e:
        logger.error(f"Erro nfce_from_qrcode: {e}", exc_info=True)
        return JSONResponse(
            content={
                "tipo_documento": "erro",
                "itens": [],
                "total_nota": None,
                "data_compra": None,
                "origem": "nfce_sp_qrcode",
                "qrcode_url": body.qrcode_url,
                "mensagem": f"Erro ao consultar NFC-e via QRCode: {str(e)}",
                "confianca": 0.0,
                "debug": {"enabled": True, "exception": str(e)},
            },
            status_code=200,
        )


@app.post("/api/ocr/comprovante")
@app.post("/api/nfce/from-image")
async def extract_comprovante(file: UploadFile = File(...)):
    try:
        if ocr_engine is None:
            raise HTTPException(status_code=500, detail="OCR Engine não inicializado")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Arquivo vazio")

        qr_data = ocr_engine.extract_qrcode(image_bytes)
        ocr_result = ocr_engine.extract_text(image_bytes)
        structured_data = ocr_engine.structure_data(ocr_result, qr_data)

        # TEMPORÁRIO: devolver linhas brutas de OCR para debug
        structured_data["ocr_raw_lines"] = ocr_result
        return JSONResponse(content=structured_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro interno: {e}", exc_info=True)
        return JSONResponse(
            content={
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": None,
                "mensagem": f"Erro interno: {str(e)}",
                "confianca": 0.0,
            },
            status_code=200,
        )


@app.post("/api/ocr/qrcode-only", response_model=QRCodeResponse)
async def extract_qrcode_only(file: UploadFile = File(...)):
    try:
        if ocr_engine is None:
            raise HTTPException(status_code=500, detail="OCR Engine não inicializado")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser imagem")

        image_bytes = await file.read()
        qr_data = ocr_engine.extract_qrcode(image_bytes)

        if not qr_data:
            return {"found": False, "data": None}
        return {"found": True, "data": qr_data[0]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro QR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, workers=2)