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
print(">>> OCR_ENGINE.PY CARREGADO (VERSÃO: CONFIANÇA MATEMÁTICA ABSOLUTA) <<<")
print("=" * 80)

logger = logging.getLogger(__name__)

class OCREngine:
    KEYWORDS_VENDA = ['recebido', 'pix recebido', 'crédito em conta', 'depósito', 'recibo']

    def __init__(self, use_gpu: bool = False):
        try:
            params = {"use_angle_cls": False, "lang": "pt", "show_log": False}
            if use_gpu: params["use_gpu"] = True
            self.ocr = PaddleOCR(**params)
            logger.info("✓ PaddleOCR inicializado")
        except Exception as e:
            try:
                logger.warning(f"Erro init padrao: {e}. Tentando init minimo.")
                self.ocr = PaddleOCR(lang='pt')
            except Exception as e2:
                logger.error(f"✗ Erro OCR: {e2}")
                raise
        self.text_processor = TextProcessor()

    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Tenta várias técnicas de imagem para achar o QR
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
            
            logger.info(f"=== OCR LINES ({len(lines)}) ===")
            for i, l in enumerate(lines): logger.info(f"{i:02d}: {l['text']}")
            logger.info("=== FIM OCR LINES ===")
            
            return lines
        except: return []

    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        full_text = '\n'.join([l['text'] for l in ocr_lines])
        tipo = 'venda' if any(k in full_text.lower() for k in self.KEYWORDS_VENDA) else 'gasto'
        
        # 1. Tenta extração via cabeçalho (01 1234...)
        itens = self._extract_items_smart(ocr_lines, full_text, tipo)
        
        # 2. Se falhar, tenta extração via "X" (1 UN x 10,00)
        if not itens:
            logger.warning("Estratégia 1 falhou. Tentando Fallback 'X'...")
            itens = self._extract_items_fallback_x(ocr_lines, full_text, tipo)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]['data'] if qr_data else None,
            "mensagem": None if itens else "Nenhum item válido encontrado",
            "confianca": 1.0 if itens else 0.0
        }

    def _extract_items_smart(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        data_compra = self._extract_date(full_text)
        itens = []
        
        # Regex simplificado: Procura numeros no inicio da linha seguidos de espaço e mais numeros
        header_regex = re.compile(r'(?:^|\s)(?:\d{1,3}|C\d)\s+\d+')
        header_indices = [i for i, l in enumerate(lines) if header_regex.search(" ".join(l['text'].split()))]
        
        logger.info(f"Indices Candidatos (Smart): {header_indices}")

        for i, idx in enumerate(header_indices):
            # Expande a área de busca (1 pra trás, 6 pra frente) para garantir que pegamos tudo
            start_search = max(0, idx - 1)
            end_search = header_indices[i+1] if i+1 < len(header_indices) else min(idx + 6, len(lines))
            if i > 0 and start_search < header_indices[i-1]: start_search = header_indices[i-1] + 1
            
            search_lines = lines[start_search:end_search]
            block_text = " ".join([l['text'] for l in search_lines])
            
            # Limpa Nome
            header_text = " ".join(lines[idx]['text'].split())
            nome = re.sub(r'^(?:\d{1,3}|C\d)\s+\d+\s+', '', header_text) # Tira codigo
            nome = re.sub(r'\s+(KG|UN|LT|L)\s*$', '', nome, flags=re.IGNORECASE) # Tira unidade fim
            match_lixo = re.search(r'\s+[\d\.\,]*\s*(UN|KG|L|LT).*?[xX]', nome, re.IGNORECASE) # Tira lixo '1 UN x'
            if match_lixo: nome = nome[:match_lixo.start()]
            nome = nome.strip(" -.'\"") or "Produto"

            item = self._parse_block_values(block_text, nome, data_compra, tipo)
            if item: 
                logger.info(f"  -> ITEM ENCONTRADO (Smart): {item['item']} | Tot: {item['valor_total']}")
                itens.append(item)
            else:
                logger.warning(f"  -> Falha ao processar bloco: {nome}")
        
        return itens

    def _extract_items_fallback_x(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        data_compra = self._extract_date(full_text)
        itens = []
        
        # Procura linhas com "NUM UN X NUM"
        x_regex = re.compile(r'([0-9]+[.,]?[0-9]*)\s*(?:UN|KG|L|LT|PC)\s*[xX]\s*([0-9]+[.,][0-9a-zA-Z]+)', re.IGNORECASE)
        x_indices = [i for i, l in enumerate(lines) if x_regex.search(l['text'])]
        
        logger.info(f"Indices Candidatos (Fallback X): {x_indices}")

        for idx in x_indices:
            # Pega contexto ao redor da linha do X
            start_search = max(0, idx - 1)
            end_search = min(idx + 2, len(lines))
            block_text = " ".join([l['text'] for l in lines[start_search:end_search]])

            # Tenta achar nome na linha anterior
            nome = "Produto"
            if idx > 0:
                prev = lines[idx-1]['text']
                if not x_regex.search(prev) and len(prev) > 3:
                     nome = re.sub(r'^(?:\d{1,3}|C\d)\s+\d+\s+', '', prev).strip(" -.'\"")

            item = self._parse_block_values(block_text, nome, data_compra, tipo)
            if item: 
                logger.info(f"  -> ITEM ENCONTRADO (Fallback): {item['item']} | Tot: {item['valor_total']}")
                itens.append(item)

        return itens

    def _parse_block_values(self, block_text, nome, data_compra, tipo):
        """Extrai valores do bloco de texto e aplica Confiança Matemática."""
        qtd = 1.0
        unit = None
        
        # Limpeza de OCR (ex: 19,9C -> 19,90)
        clean_block = block_text.replace('C', '0').replace('O', '0')
        
        # Regex Qtd x Unit
        m_full = re.search(r'([0-9]+[.,]?[0-9]*)\s*(?:UN|KG|L|LT|PC)\s*[xX]\s*([0-9]+[.,][0-9]+)', clean_block, re.IGNORECASE)
        if m_full:
            try:
                qtd = float(m_full.group(1).replace(',', '.'))
                unit = float(m_full.group(2).replace(',', '.'))
            except: pass
        else:
             # Tenta achar QTD sozinha
            m_qtd = re.search(r'([0-9]+[.,]?[0-9]*)\s*(?:UN|KG|L|LT|PC)\b', clean_block, re.IGNORECASE)
            if m_qtd: 
                try: qtd = float(m_qtd.group(1).replace(',', '.'))
                except: pass

        valor_total = None

        # --- LÓGICA DE CONFIANÇA MATEMÁTICA ---
        if unit is not None:
            # Se temos Qtd e Unit, CALCULAMOS O TOTAL.
            # Não dependemos de achá-lo no OCR. A matemática manda.
            valor_total = round(qtd * unit, 2)
        else:
            # Se não temos unitário, temos que caçar o total no texto
            candidates = []
            for val_str in re.findall(r'(\d+[.,]\d{2})', block_text):
                try: candidates.append(float(val_str.replace(',', '.')))
                except: pass
            
            # Pega o maior valor que não seja a própria quantidade
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