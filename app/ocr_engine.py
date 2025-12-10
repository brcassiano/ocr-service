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
                self._log("Erro: Imagem invalida")
                return []
            
            # Executa OCR (sem argumentos, usa config do init)
            result = self.ocr.ocr(img) 
            
            if result is None:
                self._log("OCR retornou None.")
                return []

            # === ADAPTADOR UNIVERSAL ===
            raw_lines = []

            # 1. Tenta tratar como lista padrão
            if isinstance(result, list):
                if len(result) > 0:
                    # Se for lista de listas plana
                    if isinstance(result[0], list) and len(result[0]) == 2 and isinstance(result[0][1], tuple):
                         raw_lines = result
                    # Se for lista aninhada (padrão antigo)
                    elif isinstance(result[0], list) and isinstance(result[0][0], list):
                         raw_lines = result[0]
                    # Se for lista de objetos estranhos (OCRResult)
                    else:
                         # Tenta converter o objeto container em lista, ou usa a própria lista se já for iterável
                         try:
                            # Tenta acessar result[0] se for o container
                            if hasattr(result[0], '__iter__') and not isinstance(result[0], dict):
                                 # Verifica se result[0] é o container de linhas ou uma linha
                                 # Hack: converte pra lista e vê o tamanho/tipo
                                 temp = list(result[0])
                                 if len(temp) > 0 and (isinstance(temp[0], dict) or isinstance(temp[0], list)):
                                     raw_lines = temp
                                 else:
                                     raw_lines = result
                            else:
                                raw_lines = result
                         except:
                            raw_lines = result
            else:
                 # Se result não for lista, tenta iterar ele direto
                 try:
                     raw_lines = list(result)
                 except:
                     pass

            self._log(f"Raw Lines extraidas: {len(raw_lines)}")

            # === LOG ESPIÃO (O MAIS IMPORTANTE) ===
            if len(raw_lines) > 0:
                self._log(f"DEBUG TIPO ITEM 0: {type(raw_lines[0])}")
                self._log(f"DEBUG CONTEUDO ITEM 0: {str(raw_lines[0])}")

            # === PARSER DE LINHAS ===
            lines = []
            for item in raw_lines:
                text = ""
                confidence = 0.0
                y_pos = 0
                
                # TIPO A: Lista [ [[x,y]..], (text, conf) ]
                if isinstance(item, list) and len(item) >= 2:
                    if isinstance(item[0], list) and len(item[0]) > 0:
                         y_pos = int(item[0][0][1])
                    if isinstance(item[1], tuple) or isinstance(item[1], list):
                        text = item[1][0]
                        confidence = item[1][1]
                
                # TIPO B: Dicionário (PaddleX / OCRResult convertido)
                elif isinstance(item, dict):
                    # Tenta todas as variações conhecidas de chaves
                    text = item.get('text') or item.get('rec_text') or item.get('label') or ''
                    
                    # Confiança
                    conf_val = item.get('confidence') or item.get('score') or item.get('rec_score') or 0
                    if conf_val: confidence = float(conf_val)

                    # Box/Posição
                    # Procura 'box', 'dt_boxes', 'bbox', 'poly', 'dt_polys'
                    box = item.get('box') or item.get('dt_boxes') or item.get('bbox') or item.get('poly') or item.get('dt_polys')
                    
                    if box and isinstance(box, list) and len(box) > 0:
                        # Box pode ser [[x,y],...] ou [x,y,w,h]
                        first_point = box[0]
                        if isinstance(first_point, list) or isinstance(first_point, tuple):
                             y_pos = int(first_point[1]) # [[x,y]...]
                        else:
                             # Formato [xmin, ymin, xmax, ymax]
                             y_pos = int(box[1]) 

                # TIPO C: Objeto
                elif hasattr(item, 'text') or hasattr(item, 'rec_text'):
                    text = getattr(item, 'text', '') or getattr(item, 'rec_text', '')
                    confidence = getattr(item, 'confidence', 0) or getattr(item, 'score', 0)
                    box = getattr(item, 'box', None) or getattr(item, 'dt_boxes', None) or getattr(item, 'bbox', None)
                    if box and len(box) > 0: y_pos = int(box[0][1])

                if text and confidence > 0.4:
                    lines.append({
                        'text': str(text).strip(),
                        'confidence': round(confidence, 3),
                        'y_position': y_pos
                    })

            lines.sort(key=lambda x: x['y_position'])
            
            self._log(f"Final Lines processadas: {len(lines)}")
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
