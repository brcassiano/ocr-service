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

        # 1) Parser especializado por linha de item
        itens = self._extract_items_by_line(ocr_lines, tipo)

        # 2) Fallbacks antigos (se nada foi encontrado)
        if not itens:
            itens = self._extract_items_smart(ocr_lines, full_text, tipo)
        if not itens:
            itens = self._extract_items_fallback_x(ocr_lines, full_text, tipo)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]["data"] if qr_data else None,
            "mensagem": None,
            "confianca": 1.0 if itens else 0.0,
        }



    # ---------------- NOVO PARSER POR LINHA ----------------

    def _extract_items_by_line(self, lines: List[Dict], tipo: str) -> List[Dict]:
        """
        Extrai todos os itens da NFC-e do supermercado, ignorando código de barras.

        Layout típico:
        01 07891515546335
        BATATA SADIA 1,05KG 1 UN x 15,89 T03 15,89
        """
        data_compra = self._extract_date("\n".join(l["text"] for l in lines))
        itens: List[Dict] = []

        # 1) normaliza textos em ordem
        raw = [l.get("text", "").strip() for l in lines if l.get("text")]

        # 2) agrupa blocos por item
        header_pat = re.compile(r'^\s*(\d{1,3})\s+(\d{8,14})\b')
        blocks: list[list[str]] = []
        current: list[str] = []

        for t in raw:
            if header_pat.match(t):
                if current:
                    blocks.append(current)
                current = [t]
            else:
                # só agrega em bloco se já está dentro de um item
                if current:
                    current.append(t)
        if current:
            blocks.append(current)

        # 3) regex para descrição + qtd + unitário + total
        line_regex = re.compile(
            r'^(?P<desc>.+?)\s+'                      # descrição
            r'(?P<qtd>\d+[.,]?\d*)\s+'                # quantidade
            r'(?P<un>UN|KG|L|LT|PC)\s*'               # unidade
            r'[xX]\s*'
            r'(?P<vunit>\d+[.,]\d{2})\s+'             # valor unitário
            r'(?:[A-Z0-9]{1,4}\s+)?'                  # T03 etc (opcional)
            r'(?P<vtotal>\d+[.,]\d{2})\s*$'           # valor total
        )

        for block_lines in blocks:
            # junta todas as linhas do item num único texto
            full_block = " ".join(block_lines)
            # remove o prefixo "NN CODIGO"
            block_no_header = header_pat.sub("", full_block, count=1).strip()

            # tenta casar com regex completo
            m = line_regex.match(block_no_header)
            if not m:
                # fallback: tenta usar o maior trecho como nome e último número como total
                numbers = re.findall(r'(\d+[.,]\d{2})', block_no_header)
                if not numbers:
                    continue
                vtotal = float(numbers[-1].replace(",", "."))
                vunit = float(numbers[-2].replace(",", ".")) if len(numbers) >= 2 else None
                desc = self._clean_desc(block_no_header)
                itens.append({
                    "item": desc,
                    "quantidade": 1.0,
                    "valor_unitario": vunit,
                    "valor_total": vtotal,
                    "data_compra": data_compra if tipo == "gasto" else None,
                    "data_venda": data_compra if tipo == "venda" else None,
                })
                continue

            desc = self._clean_desc(m.group("desc"))
            qtd = float(m.group("qtd").replace(",", "."))
            vunit = float(m.group("vunit").replace(",", "."))
            vtotal = float(m.group("vtotal").replace(",", "."))

            itens.append({
                "item": desc,
                "quantidade": qtd,
                "valor_unitario": vunit,
                "valor_total": vtotal,
                "data_compra": data_compra if tipo == "gasto" else None,
                "data_venda": data_compra if tipo == "venda" else None,
            })

        return itens



    def _clean_desc(self, desc: str) -> str:
        # remove múltiplos espaços
        d = re.sub(r"\s+", " ", desc).strip().upper()
        # remove qualquer sequência inicial “R$”, números soltos etc. que não façam sentido no nome
        # mas NÃO remove "1,05KG" etc.
        d = re.sub(r'^\d+\s+', '', d)
        d = re.sub(r"[^A-Z0-9À-Ü\s\-\.,/]", "", d)
        for wrong, right in self.COMMON_CORRECTIONS.items():
            if wrong in d:
                d = d.replace(wrong, right)
        return d


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