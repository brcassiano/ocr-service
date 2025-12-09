from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from .ocr_engine import OCREngine
from .models import OCRResponse, QRCodeResponse, HealthResponse
from . import __version__

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="MEIre OCR Service",
    description="Serviço de OCR especializado em comprovantes fiscais brasileiros",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar OCR Engine
ocr_engine = None

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
        "version": __version__
    }

@app.post("/api/ocr/comprovante", response_model=OCRResponse)
@app.post("/api/nfce/from-image", response_model=OCRResponse) # Alias para compatibilidade
async def extract_comprovante(file: UploadFile = File(...)):
    """
    Extrai dados estruturados de comprovantes fiscais brasileiros.
    """
    try:
        logger.info(f"Processando arquivo: {file.filename} ({file.content_type})")
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
        
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Arquivo vazio")
        
        logger.info(f"Imagem carregada: {len(image_bytes)} bytes")
        
        # 1. Tentar extrair QR Code (apenas para retornar na resposta)
        qr_data = ocr_engine.extract_qrcode(image_bytes)
        
        # 2. Executar OCR completo
        ocr_result = ocr_engine.extract_text(image_bytes)
        
        # 3. Estruturar dados (Usando a lógica MATEMÁTICA GLOBAL)
        structured_data = ocr_engine.structure_data(ocr_result, qr_data)
        
        logger.info(f"Processamento concluído: {len(structured_data['itens'])} itens")
        
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
                "mensagem": f"Erro interno: {str(e)}",
                "confianca": 0.0
            },
            status_code=200
        )

@app.post("/api/ocr/qrcode-only", response_model=QRCodeResponse)
async def extract_qrcode_only(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser imagem")
        
        image_bytes = await file.read()
        qr_data = ocr_engine.extract_qrcode(image_bytes)
        
        if not qr_data: return {"found": False, "data": None}
        return {"found": True, "data": qr_data[0]}
        
    except Exception as e:
        logger.error(f"Erro QR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, workers=2)