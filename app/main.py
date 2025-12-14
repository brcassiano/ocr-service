import logging
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .ocr_engine import OCREngine
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