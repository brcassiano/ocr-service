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
        except: return None

    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        self.debug_log = [] 
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: 
                self._log("Erro: Imagem invalida")
                return []
            
            result = self.ocr.ocr(img) 
            if result is None:
                self._log("OCR retornou None.")
                return []

            # === PARSER DE ESTRUTURA ===
            # Detecta formato PaddleX vs Lista
            first_obj = result[0] if isinstance(result, list) and len(result) > 0 else result

            def get_attr(obj, key):
                if isinstance(obj, dict): return obj.get(key)
                return getattr(obj, key, None)

            rec_texts = get_attr(first_obj, 'rec_texts')
            rec_scores = get_attr(first_obj, 'rec_scores')
            rec_polys = get_attr(first_obj, 'dt_polys') or get_attr(first_obj, 'rec_polys')

            lines = []

            # FORMATO PADDLEX (Listas Paralelas)
            if rec_texts and rec_scores and len(rec_texts) == len(rec_scores):
                self._log(f"Formato: PaddleX ({len(rec_texts)} linhas)")
                for i in range(len(rec_texts)):
                    text = rec_texts[i]
                    confidence = rec_scores[i]
                    y_pos = 0
                    if rec_polys and i < len(rec_polys):
                        poly = rec_polys[i]
                        if hasattr(poly, 'tolist'): poly = poly.tolist()
                        if len(poly) > 0:
                            if isinstance(poly[0], list) or isinstance(poly[0], tuple) or hasattr(poly[0], 'shape'):
                                y_pos = int(poly[0][1])
                            elif len(poly) >= 2:
                                y_pos = int(poly[1])
                    
                    if text and float(confidence) > 0.4:
                        lines.append({'text': str(text).strip(), 'confidence': round(float(confidence), 3), 'y_position': y_pos})

            # FORMATO LEGADO
            else:
                self._log("Formato: Legado")
                raw_lines = []
                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], list) and len(result[0]) == 2 and isinstance(result[0][1], tuple):
                         raw_lines = result
                    elif isinstance(result[0], list) and isinstance(result[0][0], list):
                         raw_lines = result[0]
                    else:
                         if not get_attr(result[0], 'rec_texts'): raw_lines = result

                for item in raw_lines:
                    text = ""
                    confidence = 0.0
                    y_pos = 0
                    if isinstance(item, list) and len(item) >= 2:
                        if isinstance(item[0], list) and len(item[0]) > 0: y_pos = int(item[0][0][1])
                        if isinstance(item[1], tuple) or isinstance(item[1], list):
                            text = item[1][0]
                            confidence = item[1][1]
                    if text and confidence > 0.4:
                        lines.append({'text': str(text).strip(), 'confidence': round(confidence, 3), 'y_position': y_pos})

            lines.sort(key=lambda x: x['y_position'])
            self._log(f"Linhas finais: {len(lines)}")
            return lines

        except Exception as e:
            self._log(f"Erro FATAL extract_text: {str(e)}")
            return []

    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        full_text = '\n'.join([l['text'] for l in ocr_lines])
        tipo = 'venda' if any(k in full_text.lower() for k in self.KEYWORDS_VENDA) else 'gasto'
        
        itens = self._extract_items_smart(ocr_lines, full_text, tipo)
        
        # Fallback apenas se Smart retornar zero itens
        if not itens:
            self._log("Smart zero itens. Tentando Fallback X...")
            itens = self._extract_items_fallback_x(ocr_lines, full_text, tipo)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]['data'] if qr_data else None,
            "mensagem": " | ".join(self.debug_log[-5:]) if not itens else None,
            "confianca": 1.0 if itens else 0.0
        }

    def _extract_items_smart(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        data_compra = self._extract_date(full_text)
        itens = []
        
        # Regex NFC-e: Inicio linha + (digitos ou C+digito) + espaco + digitos longos
        header_regex = re.compile(r'(?:^|\s)(?:\d{1,3}|C\d)\s+\d+')
        header_indices = [i for i, l in enumerate(lines) if header_regex.search(l['text'])]
        
        self._log(f"Indices: {header_indices}")

        for i, idx in enumerate(header_indices):
            # === CORREÇÃO CRÍTICA DE JANELA ===
            # Janela começa no índice ATUAL (idx). Nunca idx-1.
            # Vai até o próximo índice de cabeçalho.
            start_search = idx
            
            if i + 1 < len(header_indices):
                end_search = header_indices[i+1] # Para EXATAMENTE antes do próximo item
            else:
                end_search = min(idx + 6, len(lines)) # Último item: olha 6 linhas pra frente
            
            search_lines = lines[start_search:end_search]
            block_text = " ".join([l['text'] for l in search_lines])
            
            # Limpeza Nome
            header_text = lines[idx]['text']
            nome = re.sub(r'^(?:\d{1,3}|C\d)\s+\d+\s+', '', header_text) 
            nome = re.sub(r'\s+(KG|UN|LT|L)\s*$', '', nome, flags=re.IGNORECASE)
            # Remove lixo do final se tiver X
            match_lixo = re.search(r'\s+[\d\.\,]*\s*(UN|KG|L|LT).*?[xX]', nome, re.IGNORECASE)
            if match_lixo: nome = nome[:match_lixo.start()]
            nome = nome.strip(" -.'\"") or "Produto"

            item = self._parse_block_values(block_text, nome, data_compra, tipo)
            
            if item: 
                itens.append(item)
            else:
                # Se falhar usando a janela restrita, aí sim tentamos um "lookahead" suave
                # Caso a linha de valores tenha ficado órfã logo abaixo do limite
                if i + 1 < len(header_indices): # Só se não for o último
                     extra_lines = lines[start_search : end_search + 1] # Pega +1 linha
                     block_text_extra = " ".join([l['text'] for l in extra_lines])
                     item_retry = self._parse_block_values(block_text_extra, nome, data_compra, tipo)
                     if item_retry: itens.append(item_retry)

        return itens

    def _extract_items_fallback_x(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        data_compra = self._extract_date(full_text)
        itens = []
        
        x_regex = re.compile(r'([0-9]+[.,]?[0-9]*)\s*.*?[xX]\s*([0-9]+[.,]?[0-9a-zA-Z]+)', re.IGNORECASE)
        x_indices = [i for i, l in enumerate(lines) if x_regex.search(l['text'])]
        
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
        
        # Regex Qtd x Unit
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
            # Caça ao total
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