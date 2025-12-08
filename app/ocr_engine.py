from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Optional
import logging
from .utils import TextProcessor

logger = logging.getLogger(__name__)

class OCREngine:
    """Motor de OCR simplificado"""
    
    KEYWORDS_VENDA = [
        'recebido', 'pix recebido', 'crédito em conta', 'depósito',
        'transferência recebida', 'recibo', 'valor recebido'
    ]
    
    def __init__(self, use_gpu: bool = False):
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='pt',
                use_gpu=use_gpu,
                show_log=False
            )
            logger.info(f"PaddleOCR inicializado")
        except Exception as e:
            logger.error(f"Erro OCR: {e}")
            raise
        
        self.text_processor = TextProcessor()
    
    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        """Extrai QR Code com múltiplas tentativas"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Tentativa 1: Imagem original
            decoded = decode(gray)
            
            # Tentativa 2: Binarização
            if not decoded:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                decoded = decode(binary)
            
            # Tentativa 3: Aumentar contraste
            if not decoded:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                decoded = decode(enhanced)
            
            # Tentativa 4: Redimensionar
            if not decoded:
                height, width = gray.shape
                scale = 2.0
                resized = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
                decoded = decode(resized)
            
            if not decoded:
                logger.warning("QR Code não detectado")
                return None
            
            results = []
            for obj in decoded:
                if obj.type == 'QRCODE':
                    data = obj.data.decode('utf-8', errors='ignore')
                    results.append({'data': data, 'type': obj.type})
                    logger.info(f"✓ QR Code: {data[:60]}...")
            
            return results if results else None
            
        except Exception as e:
            logger.error(f"Erro QR: {e}")
            return None
    
    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        """OCR completo"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return []
            
            # Pré-processar
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            
            # OCR
            result = self.ocr.ocr(denoised, cls=True)
            
            if not result or not result[0]:
                return []
            
            lines = []
            for line in result[0]:
                text = line[1][0].strip()
                confidence = line[1][1]
                
                if confidence > 0.5:
                    lines.append({
                        'text': text,
                        'confidence': round(confidence, 3),
                        'y_position': int(line[0][0][1])
                    })
            
            lines.sort(key=lambda x: x['y_position'])
            logger.info(f"OCR: {len(lines)} linhas extraídas")
            
            return lines
            
        except Exception as e:
            logger.error(f"Erro OCR: {e}")
            return []
    
    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        """Estrutura dados de forma mais robusta"""
        if not ocr_lines:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": None,
                "mensagem": "Não consegui extrair texto. Tente foto mais nítida.",
                "confianca": 0.0
            }
        
        # Concatenar texto completo
        full_text = '\n'.join([line['text'] for line in ocr_lines])
        logger.info(f"Texto completo extraído:\n{full_text[:500]}")
        
        # Detectar tipo
        tipo = self._detect_type(full_text)
        
        # Extrair dados estruturados
        itens = self._extract_items_smart(ocr_lines, full_text, tipo)
        
        # QR Code
        qr_url = qr_data[0]['data'] if qr_data else None
        
        # Confiança
        avg_conf = sum(l['confidence'] for l in ocr_lines) / len(ocr_lines)
        
        if not itens:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": qr_url,
                "mensagem": "Não encontrei valores válidos na imagem.",
                "confianca": round(avg_conf, 3)
            }
        
        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_url,
            "mensagem": None,
            "confianca": round(avg_conf, 3)
        }
    
    def _detect_type(self, text: str) -> str:
        """Detecta tipo de documento"""
        text_lower = text.lower()
        
        for kw in self.KEYWORDS_VENDA:
            if kw in text_lower:
                return 'venda'
        
        return 'gasto'
    
    def _extract_items_smart(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        """
        Extração inteligente focada em cupons fiscais brasileiros
        """
        # Extrair data
        date_patterns = [
            r'emiss[aã]o:\s*(\d{2}/\d{2}/\d{4})',
            r'data:\s*(\d{2}/\d{2}/\d{4})',
            r'(\d{2}/\d{2}/\d{4})'
        ]
        
        data_compra = None
        for pattern in date_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                data_compra = match.group(1)
                break
        
        if not data_compra:
            data_compra = datetime.now().strftime('%d/%m/%Y')
        
        logger.info(f"Data extraída: {data_compra}")
        
        if tipo == 'venda':
            # Para vendas: pegar maior valor
            valores = re.findall(r'(\d+[.,]\d{2})', full_text)
            if valores:
                valor_max = max([float(v.replace(',', '.')) for v in valores])
                return [{
                    "item": None,
                    "quantidade": None,
                    "valor_unitario": None,
                    "valor_total": valor_max,
                    "data_venda": data_compra
                }]
        
        # Para GASTOS: buscar padrão de item de NFC-e
        # Formato: "01 CODIGO_BARRA NOME_PRODUTO QTDE UN x VALOR F TOTAL"
        itens = []
        
        for line in lines:
            text = line['text']
            
            # Padrão NFC-e: linha começa com 2 dígitos + código longo
            match = re.match(r'^(\d{2})\s+(\d{8,})\s+(.+)', text)
            if match:
                # Extrair valores da linha
                valores = re.findall(r'(\d+[.,]\d{2})', text)
                
                if valores:
                    # Nome do produto (parte 3 do match até os valores)
                    nome_bruto = match.group(3)
                    
                    # Limpar nome: remover valores e códigos
                    nome = nome_bruto
                    for val in valores:
                        nome = nome.replace(val, '')
                    nome = re.sub(r'\d+\s*(un|kg|lt|pc)', '', nome, flags=re.IGNORECASE)
                    nome = re.sub(r'\s+x\s+', ' ', nome)
                    nome = re.sub(r'\s+[a-z]\s+', ' ', nome)
                    nome = nome.strip()
                    
                    # Pegar último valor como total
                    valor_total = float(valores[-1].replace(',', '.'))
                    
                    # Quantidade
                    qty_match = re.search(r'(\d+)\s*un', text, re.IGNORECASE)
                    qtd = float(qty_match.group(1)) if qty_match else 1
                    
                    # Valor unitário
                    valor_unit = float(valores[-2].replace(',', '.')) if len(valores) >= 2 else valor_total
                    
                    itens.append({
                        "item": nome if len(nome) > 2 else "Produto",
                        "quantidade": qtd,
                        "valor_unitario": round(valor_unit, 2),
                        "valor_total": round(valor_total, 2),
                        "data_compra": data_compra
                    })
                    
                    logger.info(f"✓ Item: {nome} - R$ {valor_total:.2f}")
        
        # Se não encontrou nenhum item no formato padrão, pegar valores gerais
        if not itens:
            logger.warning("Formato NFC-e não detectado, extraindo valores gerais")
            
            # Procurar por TOTAL ou VALOR TOTAL
            total_match = re.search(r'(?:valor\s+)?total.*?(\d+[.,]\d{2})', full_text, re.IGNORECASE)
            if total_match:
                valor = float(total_match.group(1).replace(',', '.'))
                itens.append({
                    "item": "Total da compra",
                    "quantidade": 1,
                    "valor_unitario": valor,
                    "valor_total": valor,
                    "data_compra": data_compra
                })
        
        return itens