from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from .ocrengine import OCREngine
from .models import OCRResponse, QRCodeResponse, HealthResponse
from .nfce_parser import NfceParserSP
from . import version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MEIre OCR Service",
    description="Serviço de OCR especializado em comprovantes fiscais brasileiros",
    version=version,
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

ocrengine: OCREngine | None = None
nfce_parser = NfceParserSP()


@app.on_event("startup")
async def startup_event():
    global ocrengine
    logger.info("Iniciando MEIre OCR Service...")
    try:
        ocrengine = OCREngine(use_gpu=False)
        logger.info("OCR Engine inicializado com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao inicializar OCR Engine: {e}", exc_info=True)
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="meire-ocr-service", version=version)


@app.post("/api/nfce-from-image", response_model=OCRResponse)
async def extract_nfce_from_image(
    file: UploadFile = File(...),
    debug_ocr: bool = Query(False, description="Se true, inclui ocr_raw_lines na resposta."),
    qr_fallback: bool = Query(True, description="Se true, tenta fallback via QRCode (SEFAZ) quando OCR falhar."),
):
    try:
        if ocrengine is None:
            raise HTTPException(status_code=500, detail="OCR Engine não inicializado.")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem.")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Arquivo vazio.")

        qr_data = ocrengine.extract_qrcode(image_bytes)
        ocr_lines = ocrengine.extract_text(image_bytes)
        structured = ocrengine.structure_data(ocr_lines, qr_data)

        # debug opcional (sem quebrar schema)
        if debug_ocr:
            structured["ocr_raw_lines"] = ocr_lines
        else:
            structured["ocr_raw_lines"] = None

        # fallback via QRCode (SEFAZ-SP) se OCR vier vazio ou muito fraco
        if qr_fallback and structured.get("qrcode_url") and (not structured.get("itens")):
            try:
                parsed = await nfce_parser.parse(structured["qrcode_url"])
                # mantém o formato do OCRResponse
                structured["tipo_documento"] = parsed.get("tipo_documento", structured.get("tipo_documento", "gasto"))
                structured["itens"] = parsed.get("itens", []) or []
                structured["mensagem"] = structured["mensagem"] or "Itens obtidos via SEFAZ (fallback QRCode)."
                structured["confianca"] = 1.0 if structured["itens"] else structured.get("confianca", 0.0)
            except Exception as e:
                # fallback falhou: mantém OCR e registra
                logger.warning(f"Fallback SEFAZ falhou: {e}", exc_info=True)

        return OCRResponse(**structured)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro interno: {e}", exc_info=True)
        # mantém contrato
        return OCRResponse(
            tipo_documento="erro",
            itens=[],
            qrcode_url=None,
            mensagem=f"Erro interno: {str(e)}",
            confianca=0.0,
            ocr_raw_lines=None,
        )


@app.post("/api/ocr/qrcode-only", response_model=QRCodeResponse)
async def extract_qrcode_only(file: UploadFile = File(...)):
    try:
        if ocrengine is None:
            raise HTTPException(status_code=500, detail="OCR Engine não inicializado.")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem.")

        image_bytes = await file.read()
        qr_data = ocrengine.extract_qrcode(image_bytes)
        if not qr_data:
            return QRCodeResponse(found=False, data=None)

        return QRCodeResponse(found=True, data=qr_data[0])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro QR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, workers=2)