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
    KEYWORDS_VENDA = ['recebido', 'pix recebido', 'crédito em conta', 'depósito', 'recibo']

    def __init__(self, use_gpu: bool = False):
        self.debug_log = [] 
        try:
            params = {"use_angle_cls": False, "lang": "pt", "show_log": False}
            if use_gpu: params["use_gpu"] = True
            self.ocr = PaddleOCR(**params)
            self._log("✓ PaddleOCR inicializado")
        except Exception as e:
            try:
                self.ocr = PaddleOCR(lang='pt')
                self._log("✓ PaddleOCR inicializado (fallback)")
            except:
                raise
        self.text_processor = TextProcessor()

    def _log(self, msg):
        logger.info(msg)
        self.debug_log.append(msg)

    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
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
            
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(img)
            if data: return [{'data': data, 'type': 'QRCODE'}]
            return None
        except: 
            return None

    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        self.debug_log = [] 
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None: 
                self._log("Erro: Falha no cv2.imdecode (imagem invalida/corrompida)")
                return []
            
            self._log(f"Img Shape: {img.shape}")

            result = self.ocr.ocr(img) 
            
            self._log(f"Raw Type: {type(result)}")
            if isinstance(result, list):
                self._log(f"List Len: {len(result)}")
                if len(result) > 0:
                    self._log(f"Item[0] Type: {type(result[0])}")
                    if isinstance(result[0], list):
                        self._log(f"Item[0] Len: {len(result[0])}")

            if result is None:
                self._log("OCR retornou None direto.")
                return []

            ocr_data = []
            if isinstance(result, list):
                if len(result) > 0:
                    if result[0] is None:
                        self._log("OCR retornou [None]. Nenhum texto detectado.")
                        return []
                    
                    if isinstance(result[0], list):
                        first_inner = result[0]
                        if len(first_inner) > 0 and isinstance(first_inner[0], list) and len(first_inner[0]) == 4:
                            ocr_data = result[0]
                        else:
                            ocr_data = result
                    else:
                        ocr_data = result
                else:
                    self._log("OCR retornou lista vazia []")
                    return []
            
            lines = []
            for idx, line in enumerate(ocr_data):
                if not isinstance(line, list) or len(line) < 2:
                    continue
                
                text_info = line[1]
                if not isinstance(text_info, tuple) and not isinstance(text_info, list):
                    continue
                
                text = text_info[0].strip()
                confidence = text_info[1]
                
                box = line[0]
                y_pos = 0
                if isinstance(box, list) and len(box) > 0:
                    y_pos = int(box[0][1])

                if confidence > 0.4:
                    lines.append({
                        'text': text,
                        'confidence': round(confidence, 3),
                        'y_position': y_pos
                    })
            
            lines.sort(key=lambda x: x['y_position'])
            
            self._log(f"Linhas validas processadas: {len(lines)}")
            if len(lines) > 0:
                self._log(f"Ex L0: {lines[0]['text']}")
                
            return lines

        except Exception as e:
            self._log(f"Erro FATAL extract_text: {str(e)}")
            return []

    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        full_text = '\n'.join([l['text'] for l in ocr_lines])
        tipo = 'venda' if any(k in full_text.lower() for k in self.KEYWORDS_VENDA) else 'gasto'
        
        itens = self._extract_items_smart(ocr_lines, full_text, tipo)
        
        if not itens:
            self._log("Smart falhou. Tentando Fallback X...")
            itens = self._extract_items_fallback_x(ocr_lines, full_text, tipo)

        msg = None
        if not itens:
            msg = "DEBUG: " + " | ".join(self.debug_log)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]['data'] if qr_data else None,
            "mensagem": msg,
            "confianca": 1.0 if itens else 0.0
        }

    def _extract_items_smart(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        data_compra = self._extract_date(full_text)
        itens = []
        
        header_regex = re.compile(r'(?:^|\s)(?:\d{1,3}|C\d)\s+\d+')
        header_indices = [i for i, l in enumerate(lines) if header_regex.search(l['text'])]
        
        self._log(f"Indices Smart: {header_indices}")

        for i, idx in enumerate(header_indices):
            start_search = max(0, idx - 1)
            end_search = min(idx + 6, len(lines))
            
            search_lines = lines[start_search:end_search]
            block_text = " ".join([l['text'] for l in search_lines])
            
            header_text = lines[idx]['text']
            nome = re.sub(r'^(?:\d{1,3}|C\d)\s+\d+\s+', '', header_text) 
            nome = re.sub(r'\s+(KG|UN|LT|L)\s*$', '', nome, flags=re.IGNORECASE)
            match_lixo = re.search(r'\s+[\d\.\,]*\s*(UN|KG|L|LT).*?[xX]', nome, re.IGNORECASE)
            if match_lixo: nome = nome[:match_lixo.start()]
            nome = nome.strip(" -.'\"") or "Produto"

            item = self._parse_block_values(block_text, nome, data_compra, tipo)
            if item: itens.append(item)
        
        return itens

    def _extract_items_fallback_x(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        data_compra = self._extract_date(full_text)
        itens = []
        
        x_regex = re.compile(r'([0-9]+[.,]?[0-9]*)\s*.*?[xX]\s*([0-9]+[.,]?[0-9a-zA-Z]+)', re.IGNORECASE)
        x_indices = [i for i, l in enumerate(lines) if x_regex.search(l['text'])]
        
        self._log(f"Indices X: {x_indices}")

        for idx in x_indices:
            start_search = max(0, idx - 1)
            end_search = min(idx + 2, len(lines))
            block_text = " ".join([l['text'] for l in lines[start_search:end_search]])

            nome = "Produto"
            if idx > 0:
                prev = lines[idx-1]['text']
                if not x_regex.search(prev) and len(prev) > 3:
                    nome = re.sub(r'^(?:\d{1,3}|C\d)\s+\d+\s+', '', prev).strip(" -.'\"")

            item = self._parse_block_values(block_text, nome, data_compra, tipo)
            if item: itens.append(item)

        return itens

    def _parse_block_values(self, block_text, nome, data_compra, tipo):
        qtd = 1.0
        unit = None
        
        clean_block = block_text.replace('C', '0').replace('O', '0')
        
        m_full = re.search(r'([0-9]+[.,]?[0-9]*)\s*.*?[xX]\s*([0-9]+[.,][0-9]+)', clean_block, re.IGNORECASE)
        if m_full:
            try:
                qtd = float(m_full.group(1).replace(',', '.'))
                unit = float(m_full.group(2).replace(',', '.'))
            except: pass
        else:
            m_qtd = re.search(r'([0-9]+[.,]?[0-9]*)\s*(?:UN|KG|L|LT|PC)\b', clean_block, re.IGNORECASE)
            if m_qtd: 
                try: qtd = float(m_qtd.group(1).replace(',', '.'))
                except: pass

        valor_total = None

        if unit is not None:
            valor_total = round(qtd * unit, 2)
        else:
            candidates = []
            for val_str in re.findall(r'(\d+[.,]\d{2})', block_text):
                try: candidates.append(float(val_str.replace(',', '.')))
                except: pass
            
            valid_candidates = [c for c in candidates if abs(c - qtd) > 0.001]
            if valid_candidates: 
                valor_total = valid_candidates[-1]

        if valor_total is not None:
            return {
                "item": nome,
                "quantidade": qtd,
                "valor_unitario": unit,
                "valor_total": valor_total,
                "data_compra": data_compra if tipo == 'gasto' else None,
                "data_venda": data_compra if tipo == 'venda' else None
            }
        return None

    def _extract_date(self, text: str) -> str:
        patterns = [r'emiss[aã]o[:\s]*(\d{2}/\d{2}/\d{4})', r'(\d{2}/\d{2}/\d{4})']
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m: return m.group(1)
        return datetime.now().strftime('%d/%m/%Y')
