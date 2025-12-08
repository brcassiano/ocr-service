from pydantic import BaseModel, Field
from typing import List, Optional

class Item(BaseModel):
    item: Optional[str] = Field(None, description="Nome do produto/serviço")
    quantidade: Optional[float] = Field(None, description="Quantidade do item")
    valor_unitario: Optional[float] = Field(None, description="Valor unitário")
    valor_total: float = Field(..., description="Valor total do item")
    data_compra: Optional[str] = Field(None, description="Data da compra (DD/MM/YYYY)")
    data_venda: Optional[str] = Field(None, description="Data da venda (DD/MM/YYYY)")

class OCRResponse(BaseModel):
    tipo_documento: str = Field(..., description="Tipo: 'gasto', 'venda' ou 'erro'")
    itens: List[Item] = Field(default_factory=list)
    qrcode_url: Optional[str] = Field(None, description="URL extraída do QR Code")
    mensagem: Optional[str] = Field(None, description="Mensagem de erro ou aviso")
    confianca: Optional[float] = Field(None, description="Confiança média do OCR (0-1)")

class QRCodeResponse(BaseModel):
    found: bool = Field(..., description="Se QR Code foi encontrado")
    data: Optional[dict] = Field(None, description="Dados do QR Code")

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
