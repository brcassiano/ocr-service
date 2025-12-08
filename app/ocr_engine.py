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
    """Motor OCR com extração permissiva e logs detalhados"""
    
    KEYWORDS_VENDA = ['recebido', 'pix recebido', 'crédito em conta', 'depósito', 'recibo']

    def __init__(self, use_gpu: bool = False):
        # LOG PARA VERIFICAR QUAL CÓDIGO ESTÁ RODANDO
        logger.info("=" * 60)
        logger.info("INICIALIZANDO OCR ENGINE - VERSÃO NOVA COM LOGS DETALHADOS")
        logger.info("=" * 60)
        
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=False,
                lang='pt',
                use_gpu=use_gpu,
                show_log=False
            )
            logger.info("✓ PaddleOCR inicializado com sucesso")
        except Exception as e:
            logger.error(f"✗ Erro ao inicializar OCR: {e}")
            raise
        
        self.text_processor = TextProcessor()
    
    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        """Extrai QR Code com 4 tentativas"""
        try:
            logger.info("→ Tentando extrair QR Code...")
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("✗ Falha ao decodificar imagem")
                return None
            
            logger.info(f"  Imagem: {img.shape[1]}x{img.shape[0]} pixels")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 4 tentativas
            attempts = [
                ("grayscale normal", gray),
                ("threshold binário", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ("CLAHE contraste", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)),
                ("resize 2x", cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_CUBIC))
            ]
            
            for method, processed_img in attempts:
                logger.info(f"  Tentativa: {method}")
                decoded = decode(processed_img)
                if decoded:
                    results = []
                    for obj in decoded:
                        if obj.type == 'QRCODE':
                            data = obj.data.decode('utf-8', errors='ignore')
                            results.append({'data': data, 'type': obj.type})
                            logger.info(f"  ✓ QR Code encontrado: {data[:80]}...")
                            return results
            
            logger.warning("✗ QR Code não detectado após 4 tentativas")
            return None
            
        except Exception as e:
            logger.error(f"✗ Erro ao extrair QR Code: {e}")
            return None
    
    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        """OCR completo"""
        try:
            logger.info("→ Executando OCR...")
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("✗ Falha ao decodificar imagem")
                return []
            
            # Pré-processar
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            
            result = self.ocr.ocr(denoised, cls=True)
            
            if not result or not result[0]:
                logger.warning("✗ OCR não retornou resultados")
                return []
            
            lines = []
            for line in result[0]:
                text = line[1][0].strip()
                confidence = line[1][1]
                
                if confidence > 0.4:
                    lines.append({
                        'text': text,
                        'confidence': round(confidence, 3),
                        'y_position': int(line[0][0][1])
                    })
                    logger.info(f"  [{confidence:.0%}] {text}")
            
            lines.sort(key=lambda x: x['y_position'])
            logger.info(f"✓ OCR concluído: {len(lines)} linhas extraídas")
            
            return lines
            
        except Exception as e:
            logger.error(f"✗ Erro no OCR: {e}")
            return []
    
    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        """Estrutura dados"""
        logger.info("→ Estruturando dados...")
        
        if not ocr_lines:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": None,
                "mensagem": "Não consegui extrair texto.",
                "confianca": 0.0
            }
        
        full_text = '\n'.join([line['text'] for line in ocr_lines])
        logger.info(f"=== TEXTO COMPLETO ({len(ocr_lines)} linhas) ===")
        for idx, line in enumerate(ocr_lines[:15], 1):
            logger.info(f"  {idx}. {line['text']}")
        logger.info("=== FIM TEXTO ===")
        
        tipo = self._detect_type(full_text)
        logger.info(f"  Tipo detectado: {tipo}")
        
        itens = self._extract_items_permissive(ocr_lines, full_text, tipo)
        qr_url = qr_data[0]['data'] if qr_data else None
        
        avg_conf = sum(l['confidence'] for l in ocr_lines) / len(ocr_lines)
        
        if not itens:
            logger.error("✗ Nenhum item extraído!")
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": qr_url,
                "mensagem": "Consegui ler mas não encontrei valores. Tente foto mais nítida.",
                "confianca": round(avg_conf, 3)
            }
        
        logger.info(f"✓ {len(itens)} itens extraídos com sucesso")
        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_url,
            "mensagem": None,
            "confianca": round(avg_conf, 3)
        }
    
    def _detect_type(self, text: str) -> str:
        """Detecta tipo"""
        text_lower = text.lower()
        for kw in self.KEYWORDS_VENDA:
            if kw in text_lower:
                logger.info(f"  Keyword venda: '{kw}'")
                return 'venda'
        return 'gasto'
    
    def _extract_items_permissive(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        """Extração permissiva"""
        logger.info("→ Extraindo itens (modo permissivo)...")
        
        # 1. BUSCAR DATA
        data_compra = self._extract_date(full_text)
        logger.info(f"  Data: {data_compra}")
        
        # 2. EXTRAIR VALORES
        all_valores = []
        for line in lines:
            valores_linha = re.findall(r'(\d+[.,]\d{2})', line['text'])
            for val in valores_linha:
                try:
                    val_float = float(val.replace(',', '.'))
                    if 0.01 <= val_float <= 999999:
                        all_valores.append({
                            'valor': val_float,
                            'linha': line['text']
                        })
                        logger.info(f"  Valor: R$ {val_float:.2f} em '{line['text'][:60]}'")
                except:
                    pass
        
        if not all_valores:
            logger.error("✗ Nenhum valor monetário encontrado!")
            return []
        
        logger.info(f"  Total de valores encontrados: {len(all_valores)}")
        
        # 3. VENDAS
        if tipo == 'venda':
            maior = max(all_valores, key=lambda x: x['valor'])
            logger.info(f"  Venda: R$ {maior['valor']:.2f}")
            return [{
                "item": None,
                "quantidade": None,
                "valor_unitario": None,
                "valor_total": maior['valor'],
                "data_venda": data_compra
            }]
        
        # 4. GASTOS - Estratégia 1: Linhas NFC-e
        itens = []
        for line in lines:
            text = line['text']
            
            # Formato: 01 CODIGO_LONGO PRODUTO VALORES
            if re.match(r'^\d{2}\s+\d{8,}', text):
                valores_linha = re.findall(r'(\d+[.,]\d{2})', text)
                if valores_linha:
                    nome = self._extract_product_name(text)
                    qtd = self._extract_quantity(text)
                    
                    valor_total = float(valores_linha[-1].replace(',', '.'))
                    valor_unit = float(valores_linha[-2].replace(',', '.')) if len(valores_linha) >= 2 else valor_total
                    
                    itens.append({
                        "item": nome,
                        "quantidade": qtd,
                        "valor_unitario": round(valor_unit, 2),
                        "valor_total": round(valor_total, 2),
                        "data_compra": data_compra
                    })
                    logger.info(f"  ✓ Item NFC-e: {nome} - R$ {valor_total:.2f}")
        
        # Estratégia 2: Buscar TOTAL
        if not itens:
            logger.info("  Fallback: buscando linha TOTAL...")
            total_valor = None
            
            for line in lines:
                if re.search(r'(total|pix)\s*\(?r?\$?\)?\s*(\d+[.,]\d{2})', line['text'], re.IGNORECASE):
                    match = re.search(r'(\d+[.,]\d{2})', line['text'])
                    if match:
                        total_valor = float(match.group(1).replace(',', '.'))
                        logger.info(f"  ✓ Total: R$ {total_valor:.2f}")
                        break
            
            if not total_valor:
                total_valor = max([v['valor'] for v in all_valores])
                logger.info(f"  Usando maior valor: R$ {total_valor:.2f}")
            
            itens.append({
                "item": "Compra",
                "quantidade": 1,
                "valor_unitario": total_valor,
                "valor_total": total_valor,
                "data_compra": data_compra
            })
        
        return itens
    
    def _extract_date(self, text: str) -> str:
        """Extrai data"""
        patterns = [
            r'emiss[aã]o[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'data[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'(\d{2}/\d{2}/\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1).replace('-', '/')
                logger.info(f"  Data encontrada: {date_str}")
                return date_str
        
        hoje = datetime.now().strftime('%d/%m/%Y')
        logger.warning(f"  Data não encontrada, usando hoje: {hoje}")
        return hoje
    
    def _extract_product_name(self, text: str) -> str:
        """Extrai nome do produto"""
        text = re.sub(r'^\d{2}\s+\d{8,}\s*', '', text)
        text = re.sub(r'\d+[.,]\d{2}', '', text)
        text = re.sub(r'\d+\s*(un|kg|lt|pc)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+[a-z]\s+', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+x\s+', ' ', text)
        nome = text.strip()
        return nome if len(nome) > 2 else "Produto"
    
    def _extract_quantity(self, text: str) -> float:
        """Extrai quantidade"""
        match = re.search(r'(\d+)\s*un', text, re.IGNORECASE)
        return float(match.group(1)) if match else 1.0