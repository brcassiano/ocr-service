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

    COMMON_CORRECTIONS = {
        "ALHOTRADIC": "ALHO TRADIC",
        "QJ": "QUEIJO",
        "BATATA LAVADA": "BATATA LAVADA",
    }

    def __init__(self, use_gpu: bool = False):
        self.debug_log: List[str] = []
        try:
            params = {"use_angle_cls": False, "lang": "pt", "show_log": False}
            if use_gpu:
                params["use_gpu"] = True
            self.ocr = PaddleOCR(**params)
            self._log("✓ PaddleOCR inicializado")
        except Exception:
            # fallback
            self.ocr = PaddleOCR(lang="pt")
            self._log("✓ PaddleOCR inicializado (fallback)")
        self.text_processor = TextProcessor()

    def _log(self, msg: str):
        logger.info(msg)
        self.debug_log.append(msg)

    # ---------------- QR CODE ----------------

    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for _, processed in [
                ("gray", gray),
                (
                    "thresh",
                    cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                ),
                ("clahe", cv2.createCLAHE(3.0, (8, 8)).apply(gray)),
                ("zoom", cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
            ]:
                decoded = decode(processed)
                if decoded:
                    for obj in decoded:
                        if obj.type == "QRCODE":
                            return [
                                {"data": obj.data.decode("utf-8", errors="ignore"), "type": obj.type}
                            ]

            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(img)
            if data:
                return [{"data": data, "type": "QRCODE"}]
            return None
        except Exception:
            return None

    # ---------------- OCR LINES ----------------

    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        self.debug_log = []
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return []

            result = self.ocr.ocr(img)
            if result is None:
                return []

            first_obj = result[0] if isinstance(result, list) and len(result) > 0 else result

            def get_attr(obj, key):
                if isinstance(obj, dict):
                    return obj.get(key)
                return getattr(obj, key, None)

            rec_texts = get_attr(first_obj, "rec_texts")
            rec_scores = get_attr(first_obj, "rec_scores")
            rec_polys = get_attr(first_obj, "dt_polys") or get_attr(first_obj, "rec_polys")

            lines: List[Dict] = []

            if rec_texts and rec_scores and len(rec_texts) == len(rec_scores):
                for i in range(len(rec_texts)):
                    text = rec_texts[i]
                    confidence = rec_scores[i]
                    y_pos = 0
                    if rec_polys and i < len(rec_polys):
                        poly = rec_polys[i]
                        if hasattr(poly, "tolist"):
                            poly = poly.tolist()
                        if len(poly) > 0:
                            if isinstance(poly[0], (list, tuple)):
                                y_pos = int(poly[0][1])
                            elif len(poly) >= 2:
                                y_pos = int(poly[1])
                    if text and float(confidence) > 0.4:
                        lines.append(
                            {
                                "text": str(text).strip(),
                                "confidence": round(float(confidence), 3),
                                "y_position": y_pos,
                            }
                        )
            else:
                # Fallback legado
                raw_lines = (
                    result[0]
                    if isinstance(result, list)
                    and isinstance(result[0], list)
                    and isinstance(result[0][0], list)
                    else result
                )
                for item in raw_lines:
                    text = ""
                    confidence = 0.0
                    y_pos = 0
                    if isinstance(item, list) and len(item) >= 2:
                        if isinstance(item[0], list):
                            y_pos = int(item[0][0][1])
                        if isinstance(item[1], (tuple, list)):
                            text = item[1][0]
                            confidence = item[1][1]
                    if text and confidence > 0.4:
                        lines.append(
                            {
                                "text": str(text).strip(),
                                "confidence": round(confidence, 3),
                                "y_position": y_pos,
                            }
                        )

            lines.sort(key=lambda x: x["y_position"])
            return lines
        except Exception:
            return []

    # ---------------- STRUCTURE DATA ----------------

    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        full_text = "\n".join([l["text"] for l in ocr_lines])
        tipo = "venda" if any(k in full_text.lower() for k in self.KEYWORDS_VENDA) else "gasto"

        itens = self._extract_items_by_line(ocr_lines, tipo)  # PRIMEIRO este

        if len(itens) < 3:  # só fallback se muito poucos itens
            itens_fallback = self._extract_items_smart(ocr_lines, full_text, tipo)
            itens = itens + itens_fallback[:5]  # adiciona até 5 extras

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]["data"] if qr_data else None,
            "mensagem": None,
            "confianca": 1.0 if itens else 0.0,
        }



    # ---------------- NOVO PARSER POR LINHA ----------------

    def _extract_items_by_line(self, lines: List[Dict], tipo: str) -> List[Dict]:
        data_compra = self._extract_date("\n".join(l["text"] for l in lines))
        itens: List[Dict] = []

        # 1) Ordena por altura Y
        lines_sorted = sorted(lines, key=lambda x: x.get("y_position", 0))

        # 2) Ignora linhas de footer
        footer_keywords = ["VALOR TOTAL", "QTD. TOTAL", "CARTAO", "CONSUMIDOR", "HTTPS", "PROTOCOLO"]
        valid_lines = [l for l in lines_sorted if not any(k in l.get("text", "").upper() for k in footer_keywords)]

        # 3) Agrupamento robusto por item (busca padrões "NN CODIGO" e agrupa próximas linhas)
        i = 0
        while i < len(valid_lines):
            line_text = valid_lines[i].get("text", "").strip().upper()

            # Padrão de início de item: "01 0789...", "02 0789..." etc.
            header_match = re.match(r'^\s*(\d{1,2})\s+(\d{12,14})', line_text)
            if not header_match:
                i += 1
                continue

            item_num = header_match.group(1)
            # Começa novo bloco
            block_lines = [valid_lines[i]]
            y_base = valid_lines[i].get("y_position", 0)

            # Agrupa linhas próximas (mesma altura +/- 5px)
            j = i + 1
            while j < len(valid_lines) and abs(valid_lines[j].get("y_position", 0) - y_base) <= 10:
                block_lines.append(valid_lines[j])
                j += 1

            # Monta texto completo do bloco
            full_block = " ".join([l.get("text", "") for l in block_lines]).strip()

            # 4) Remove prefixo "NN CODIGO" e parse
            block_clean = re.sub(r'^\s*\d{1,2}\s+\d{12,14}\s*', '', full_block)

            # Regex flexível para descrição + QTD UN x V_UNIT TOTAL
            parse_match = re.search(
                r'(.+?)\s*'                          # descrição
                r'(\d+[.,]?\d*)\s*'                  # qtd
                r'(UN|KG|LT|PC)\s*'                  # un
                r'[xX]?\s*'
                r'(\d+[.,]\d{2})\s*'                 # v_unit (opcional antes)
                r'(\d+[.,]\d{2})'                    # v_total (último com 2 decimais)
            , block_clean, re.IGNORECASE)

            if parse_match:
                desc, qtd, un, v_unit, v_total = parse_match.groups()
                desc = self._clean_desc(desc)
                try:
                    itens.append({
                        "item": desc,
                        "quantidade": float(qtd.replace(",", ".")),
                        "valor_unitario": float(v_unit.replace(",", ".")) if v_unit else None,
                        "valor_total": float(v_total.replace(",", ".")),
                        "data_compra": data_compra if tipo == "gasto" else None,
                        "data_venda": data_compra if tipo == "venda" else None,
                    })
                except ValueError:
                    pass  # pula itens malformados

            i = j

        return itens



    def _clean_desc(self, desc: str) -> str:
        if not desc:
            return "ITEM DESCONHECIDO"
        
        # Remove prefixos/sufixos numéricos/unidades do nome
        desc = re.sub(r'^\d+\s*', '', desc)  # remove qtd inicial
        desc = re.sub(r'\s+(UN|KG|LT|PC|L|M)\s*$', '', desc, flags=re.IGNORECASE)  # un final
        desc = re.sub(r'\s*[xX]\s*\d+[.,]\d{2}.*$', '', desc)  # ignora "x 15,89" do final
        desc = re.sub(r'\s*(T\d{2,3}|F)\s*\d+[.,]\d{2}.*$', '', desc)  # tributos T03 etc.
        
        # Padroniza
        desc = re.sub(r'\s+', ' ', desc).strip().upper()
        desc = re.sub(r'[^A-Z0-9À-Ü\s\-\.,/]', '', desc)
        
        # Correções manuais
        for wrong, right in self.COMMON_CORRECTIONS.items():
            if wrong in desc:
                desc = desc.replace(wrong, right)
        
        return desc if desc else "ITEM DESCONHECIDO"


    # ---------------- PARSERS ANTIGOS (mantidos para fallback) ----------------

    def _extract_items_smart(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        # seu código atual aqui, inalterado
        ...

    def _extract_items_fallback_x(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        # seu código atual aqui, inalterado
        ...

    def _clean_name(self, text: str) -> str:
        # seu código atual aqui, inalterado
        ...

    def _parse_block_values(self, block_text, nome, data_compra, tipo):
        # seu código atual aqui, inalterado
        ...

    def _extract_date(self, text: str) -> str:
        patterns = [r'emiss[aã]o[:\s]*(\d{2}/\d{2}/\d{4})', r'(\d{2}/\d{2}/\d{4})']
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return datetime.now().strftime('%d/%m/%Y')