from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class OCRRawLine(BaseModel):
    text: str = Field(..., description="Texto reconhecido pelo OCR.")
    confidence: float = Field(..., description="Confiança do OCR (0-1).")
    y_position: int = Field(..., description="Posição vertical aproximada do texto (pixels).")
    x_position: Optional[int] = Field(None, description="Posição horizontal aproximada do texto (pixels).")


class Item(BaseModel):
    item: Optional[str] = Field(None, description="Nome do produto/serviço.")
    quantidade: Optional[float] = Field(None, description="Quantidade do item.")
    valor_unitario: Optional[float] = Field(None, description="Valor unitário.")
    valor_total: Optional[float] = Field(None, description="Valor total do item.")
    data_compra: Optional[str] = Field(None, description="Data da compra (DD/MM/AAAA).")
    data_venda: Optional[str] = Field(None, description="Data da venda (DD/MM/AAAA).")


class OCRResponse(BaseModel):
    tipo_documento: str = Field(..., description="Tipo: gasto, venda ou erro.")
    itens: List[Item] = Field(default_factory=list, description="Lista de itens encontrados.")
    qrcode_url: Optional[str] = Field(None, description="URL extraída do QRCode (quando existir).")
    mensagem: Optional[str] = Field(None, description="Mensagem de erro/aviso.")
    confianca: Optional[float] = Field(None, description="Confiança do parser (0-1).")
    ocr_raw_lines: Optional[List[OCRRawLine]] = Field(
        None,
        description="Linhas brutas do OCR (somente para debug)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tipo_documento": "gasto",
                    "itens": [
                        {
                            "item": "BATATA SADIA 1,05KG",
                            "quantidade": 1,
                            "valor_unitario": 15.89,
                            "valor_total": 15.89,
                            "data_compra": "11/12/2025",
                            "data_venda": None
                        }
                    ],
                    "qrcode_url": "https://www.nfce.fazenda.sp.gov.br/NFCeConsultaPublica/Paginas/ConsultaQRCode.aspx?p=...",
                    "mensagem": None,
                    "confianca": 1.0,
                    "ocr_raw_lines": [
                        {"text": "01 07891515546335", "confidence": 1.0, "y_position": 380, "x_position": 120}
                    ]
                }
            ]
        }
    }


class QRCodeResponse(BaseModel):
    found: bool = Field(..., description="Se QR Code foi encontrado.")
    data: Optional[Dict[str, Any]] = Field(None, description="Dados do QR Code.")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str