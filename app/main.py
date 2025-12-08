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

# CORS (ajustar origins conforme necessário)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar OCR Engine (uma única vez, reutilizar)
ocr_engine = None

@app.on_event("startup")
async def startup_event():
    """Inicializa recursos na startup"""
    global ocr_engine
    logger.info("Iniciando MEIre OCR Service...")
    try:
        ocr_engine = OCREngine(use_gpu=False)  # Mudar para True se tiver GPU
        logger.info("OCR Engine inicializado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao inicializar OCR Engine: {e}")
        raise

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "meire-ocr-service",
        "version": __version__
    }

@app.post("/api/ocr/comprovante", response_model=OCRResponse)
async def extract_comprovante(file: UploadFile = File(...)):
    """
    Extrai dados estruturados de comprovantes fiscais brasileiros.
    
    Suporta:
    - Cupons fiscais (NFC-e) com ou sem QR Code
    - Recibos manuscritos
    - Comprovantes de PIX/transferência
    - Notas fiscais
    
    Retorna JSON estruturado com itens, valores, datas e classificação (gasto/venda).
    """
    try:
        logger.info(f"Processando arquivo: {file.filename} ({file.content_type})")
        
        # Validar tipo de arquivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Arquivo deve ser uma imagem (JPEG, PNG, etc)"
            )
        
        # Ler bytes da imagem
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Arquivo vazio")
        
        logger.info(f"Imagem carregada: {len(image_bytes)} bytes")
        
        # 1. Tentar extrair QR Code (rápido)
        qr_data = ocr_engine.extract_qrcode(image_bytes)
        
        # 2. Executar OCR completo
        ocr_result = ocr_engine.extract_text(image_bytes)
        
        # 3. Estruturar dados
        structured_data = ocr_engine.structure_data(ocr_result, qr_data)
        
        logger.info(f"Processamento concluído: tipo={structured_data['tipo_documento']}, itens={len(structured_data['itens'])}")
        
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
                "confianca": 0.0
            },
            status_code=200  # 200 para não quebrar workflow do n8n
        )

@app.post("/api/ocr/qrcode-only", response_model=QRCodeResponse)
async def extract_qrcode_only(file: UploadFile = File(...)):
    """
    Extrai apenas QR Code da imagem (mais rápido que OCR completo).
    
    Útil quando você sabe que a imagem contém apenas QR Code
    e não precisa dos dados detalhados do comprovante.
    """
    try:
        logger.info(f"Extraindo QR Code de: {file.filename}")
        
        # Validar tipo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Arquivo deve ser uma imagem"
            )
        
        # Ler imagem
        image_bytes = await file.read()
        
        # Extrair QR Code
        qr_data = ocr_engine.extract_qrcode(image_bytes)
        
        if not qr_data:
            return {"found": False, "data": None}
        
        logger.info(f"QR Code extraído com sucesso")
        return {"found": True, "data": qr_data[0]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao extrair QR Code: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # True apenas em desenvolvimento
        workers=2
    )