from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class OCRRawLine(BaseModel):
    text: str = Field(..., description="Texto reconhecido pelo OCR.")
    confidence: float = Field(..., description="Confiança do OCR (0-1).")
    y_position: int = Field(..., description="Posição vertical aproximada (pixels).")
    x_position: Optional[int] = Field(None, description="Posição horizontal aproximada (pixels).")


class Item(BaseModel):
    item: Optional[str] = Field(None, description="Nome do produto/serviço")
    quantidade: Optional[float] = Field(None, description="Quantidade do item")
    valor_unitario: Optional[float] = Field(None, description="Valor unitário")
    valor_total: Optional[float] = Field(None, description="Valor total do item")
    data_compra: Optional[str] = Field(None, description="Data da compra (DD/MM/AAAA)")
    data_venda: Optional[str] = Field(None, description="Data da venda (DD/MM/AAAA)")


class OCRResponse(BaseModel):
    tipo_documento: str = Field(..., description="Tipo: 'gasto', 'venda' ou 'erro'")
    itens: List[Item] = Field(default_factory=list)
    qrcode_url: Optional[str] = Field(None, description="URL extraída do QR Code")
    mensagem: Optional[str] = Field(None, description="Mensagem de erro ou aviso")
    confianca: Optional[float] = Field(None, description="Confiança média do OCR (0-1)")
    ocr_raw_lines: Optional[List[OCRRawLine]] = Field(None, description="Linhas brutas do OCR (debug)")


class QRCodeResponse(BaseModel):
    found: bool = Field(..., description="Se QR Code foi encontrado")
    data: Optional[Dict[str, Any]] = Field(None, description="Dados do QR Code")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str