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
    KEYWORDS_VENDA = ["recebido", "pix recebido", "crédito em conta", "depósito", "recibo"]

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
                ("thresh", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ("clahe", cv2.createCLAHE(3.0, (8, 8)).apply(gray)),
                ("zoom", cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
            ]:
                decoded = decode(processed)
                if decoded:
                    for obj in decoded:
                        if obj.type == "QRCODE":
                            return [
                                {
                                    "data": obj.data.decode("utf-8", errors="ignore"),
                                    "type": obj.type,
                                }
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
            for l in lines:
                self._log(f"OCR_LINE: y={l['y_position']} | {l['text']}")
            return lines
        except Exception:
            return []

    # ---------------- STRUCTURE DATA ----------------

    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        if not ocr_lines:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": qr_data[0]["data"] if qr_data else None,
                "mensagem": "Nenhuma linha OCR encontrada",
                "confianca": 0.0,
            }

        full_text = "\n".join([l.get("text", "") for l in ocr_lines])
        tipo = "venda" if any(k in full_text.lower() for k in self.KEYWORDS_VENDA) else "gasto"

        itens = self._extract_items_by_line(ocr_lines, tipo)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]["data"] if qr_data else None,
            "mensagem": None,
            "confianca": 1.0 if itens else 0.0,
        }

    # ---------------- PARSER POR LINHA (NFC-e) ----------------

    def _extract_items_by_line(self, lines: List[Dict], tipo: str) -> List[Dict]:
        data_compra = self._extract_date("\n".join(l.get("text", "") for l in lines))
        itens: List[Dict] = []

        lines_sorted = sorted(
            [l for l in lines if isinstance(l, dict) and "text" in l],
            key=lambda x: x.get("y_position", 0),
        )

        footer_keywords = ["VALOR TOTAL", "QTD. TOTAL", "CARTAO", "CONSUMIDOR", "HTTPS", "PROTOCOLO"]
        valid_lines = [
            l for l in lines_sorted if not any(k in l.get("text", "").upper() for k in footer_keywords)
        ]

        i = 0
        while i < len(valid_lines):
            text = valid_lines[i].get("text", "").strip().upper()

            header = re.match(r"^\s*(\d{1,2})\s+(\d{8,14})", text)
            if not header:
                i += 1
                continue

            y_base = valid_lines[i].get("y_position", 0)
            block = [valid_lines[i]]
            j = i + 1
            while j < len(valid_lines) and abs(valid_lines[j].get("y_position", 0) - y_base) <= 12:
                block.append(valid_lines[j])
                j += 1

            full_block = " ".join(b.get("text", "") for b in block)
            block_clean = re.sub(r"^\s*\d{1,2}\s+\d{8,14}\s*", "", full_block).strip()

            # desc + qtd + UN + x + v_unit + total (Txx opcional)
            m = re.search(
                r"(.+?)\s+(\d+[.,]?\d*)\s+(UN|KG|LT|PC)\s*[xX]?\s*(\d+[.,]\d{2})\s+(?:[A-Z0-9]{1,4}\s+)?(\d+[.,]\d{2})",
                block_clean,
                re.IGNORECASE,
            )

            if m:
                desc_raw, qtd, un, v_unit, v_total = m.groups()
                desc = self._clean_desc(desc_raw)
                try:
                    itens.append(
                        {
                            "item": desc,
                            "quantidade": float(qtd.replace(",", ".")),
                            "valor_unitario": float(v_unit.replace(",", ".")),
                            "valor_total": float(v_total.replace(",", ".")),
                            "data_compra": data_compra if tipo == "gasto" else None,
                            "data_venda": data_compra if tipo == "venda" else None,
                        }
                    )
                except ValueError:
                    pass

            i = j

        return itens

    def _clean_desc(self, desc: str) -> str:
        if not desc:
            return "ITEM DESCONHECIDO"

        desc = re.sub(r"^\d+\s*", "", desc)
        desc = re.sub(r"\s+(UN|KG|LT|PC|L|M)\s*$", "", desc, flags=re.IGNORECASE)
        desc = re.sub(r"\s*[xX]\s*\d+[.,]\d{2}.*$", "", desc)
        desc = re.sub(r"\s*(T\d{2,3}|F)\s*\d+[.,]\d{2}.*$", "", desc)

        desc = re.sub(r"\s+", " ", desc).strip().upper()
        desc = re.sub(r"[^A-Z0-9À-Ü\s\-\.,/]", "", desc)

        for wrong, right in self.COMMON_CORRECTIONS.items():
            if wrong in desc:
                desc = desc.replace(wrong, right)

        return desc if desc else "ITEM DESCONHECIDO"

    # ---------------- DATA ----------------

    def _extract_date(self, text: str) -> str:
        patterns = [r"emiss[aã]o[:\s]*(\d{2}/\d{2}/\d{4})", r"(\d{2}/\d{2}/\d{4})"]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return datetime.now().strftime("%d/%m/%Y")