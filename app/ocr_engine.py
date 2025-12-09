from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Optional
import logging
from .utils import TextProcessor

print("=" * 80)
print(">>> OCR_ENGINE.PY CARREGADO (VERSÃO FINAL: MATEMÁTICA GLOBAL) <<<")
print("=" * 80)

logger = logging.getLogger(__name__)

class OCREngine:
    KEYWORDS_VENDA = ['recebido', 'pix recebido', 'crédito em conta', 'depósito', 'recibo']

    def __init__(self, use_gpu: bool = False):
        try:
            self.ocr = PaddleOCR(use_angle_cls=False, lang='pt', use_gpu=use_gpu, show_log=False)
            logger.info("✓ PaddleOCR inicializado")
        except Exception as e:
            logger.error(f"✗ Erro OCR: {e}")
            raise
        self.text_processor = TextProcessor()

    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Tenta pyzbar
            for _, processed in [
                ("gray", gray),
                ("thresh", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ("clahe", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)),
                ("zoom", cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            ]:
                decoded = decode(processed)
                if decoded:
                    for obj in decoded:
                        if obj.type == 'QRCODE':
                            return [{'data': obj.data.decode('utf-8', errors='ignore'), 'type': obj.type}]
            
            # Fallback OpenCV
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(img)
            if data: return [{'data': data, 'type': 'QRCODE'}]
            return None
        except: return None

    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: return []
            result = self.ocr.ocr(img, cls=True)
            if not result or not result[0]: return []
            
            lines = []
            for line in result[0]:
                if line[1][1] > 0.4:
                    lines.append({'text': line[1][0].strip(), 'confidence': line[1][1], 'y_position': int(line[0][0][1])})
            lines.sort(key=lambda x: x['y_position'])
            
            # DEBUG
            print("=== DEBUG OCR LINES ===")
            for i, l in enumerate(lines): print(f"{i}: {l['text']}")
            print("=== FIM DEBUG ===")
            
            return lines
        except: return []

    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        full_text = '\n'.join([l['text'] for l in ocr_lines])
        tipo = 'venda' if any(k in full_text.lower() for k in self.KEYWORDS_VENDA) else 'gasto'
        
        itens = self._extract_items_smart(ocr_lines, full_text, tipo)
        
        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]['data'] if qr_data else None,
            "mensagem": None if itens else "Nenhum item válido encontrado",
            "confianca": 1.0 if itens else 0.0
        }

    def _extract_items_smart(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        # Data
        data_compra = None
        m_data = re.search(r'Emiss[aã]o[:\s]*(\d{2}/\d{2}/\d{4})|(\d{2}/\d{2}/\d{4})', full_text, re.IGNORECASE)
        if m_data: data_compra = m_data.group(1) or m_data.group(2)
        else: data_compra = datetime.now().strftime('%d/%m/%Y')

        itens = []
        
        # 1. Identifica índices dos cabeçalhos
        header_regex = re.compile(r'^(?:\d{1,2}|C\d)\s+\d{4,}\b')
        header_indices = [i for i, l in enumerate(lines) if header_regex.match(" ".join(l['text'].split()))]

        if not header_indices:
            # Fallback simples se não achar padrão NFC-e
            return []

        # 2. Processa cada item
        for i, idx in enumerate(header_indices):
            # Define "zona de busca" expandida (olha 1 linha pra trás e 4 pra frente)
            # Isso resolve o problema do "Total" aparecer antes do "Item"
            start_search = max(0, idx - 1)
            end_search = header_indices[i+1] if i+1 < len(header_indices) else min(idx + 5, len(lines))
            
            # Mas cuidado para não pegar dados do item anterior
            if i > 0 and start_search < header_indices[i-1]:
                start_search = header_indices[i-1] + 1
            
            search_lines = lines[start_search:end_search]
            block_text = " ".join([l['text'] for l in search_lines])
            
            # --- Extração de Nome ---
            header_text = " ".join(lines[idx]['text'].split())
            # Remove código inicial
            nome = re.sub(r'^(?:\d{1,2}|C\d)\s+\d+\s+', '', header_text)
            # Remove unidades no fim
            nome = re.sub(r'\s+(KG|UN|LT|L)\s*$', '', nome, flags=re.IGNORECASE)
            # Remove lixo "1 UN x..." do final do nome (Regex Agressivo)
            match_lixo = re.search(r'\s+[\d\.\,]*\s*(UN|KG|L|LT).*?[xX]', nome, re.IGNORECASE)
            if match_lixo: nome = nome[:match_lixo.start()]
            nome = nome.strip(" -.'\"") or "Produto"

            # --- Extração Qtd e Unitário ---
            qtd = 1.0
            unit = None
            
            # Procura padrao "QTD UN x UNIT"
            # Tratamento especial para "19,9C" -> "19,90"
            clean_block = block_text.replace('C', '0').replace('O', '0')
            
            m_full = re.search(r'([0-9]+[.,]?[0-9]*)\s*(?:UN|KG|L|LT)\s*[xX]\s*([0-9]+[.,][0-9]+)', clean_block, re.IGNORECASE)
            if m_full:
                try:
                    qtd = float(m_full.group(1).replace(',', '.'))
                    unit = float(m_full.group(2).replace(',', '.'))
                except: pass
            else:
                # Tenta só QTD
                m_qtd = re.search(r'([0-9]+[.,]?[0-9]*)\s*(?:UN|KG|L|LT)\b', clean_block, re.IGNORECASE)
                if m_qtd: 
                    try: qtd = float(m_qtd.group(1).replace(',', '.'))
                    except: pass

            # --- Caça ao Tesouro (Valor Total) ---
            # Coleta todos os números float da zona de busca
            candidates = []
            for val_str in re.findall(r'(\d+[.,]\d{2})', block_text):
                try: candidates.append(float(val_str.replace(',', '.')))
                except: pass
            
            valor_total = None

            if unit is not None:
                expected = round(qtd * unit, 2)
                # Procura número igual ao esperado (margem 0.05)
                # Prioriza números que NÃO sejam iguais à Qtd ou Unit
                matches = [c for c in candidates if abs(c - expected) <= 0.05]
                if matches:
                    valor_total = matches[0]
                else:
                    # Se não achou na imagem, mas temos certeza do unitário, confiamos na conta
                    valor_total = expected
                    logger.info(f"  Item {nome}: Total calculado matematicamente ({valor_total})")
            else:
                # Sem unitário, tenta o último candidato válido
                # Filtra candidatos que são iguais à Qtd (erro comum da batata 0.485)
                valid_candidates = [c for c in candidates if abs(c - qtd) > 0.001]
                if valid_candidates:
                    valor_total = valid_candidates[-1]

            if valor_total is not None:
                itens.append({
                    "item": nome,
                    "quantidade": qtd,
                    "valor_unitario": unit,
                    "valor_total": valor_total,
                    "data_compra": data_compra if tipo == 'gasto' else None,
                    "data_venda": data_compra if tipo == 'venda' else None
                })
        
        return itens