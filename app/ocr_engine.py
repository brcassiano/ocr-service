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
    }

    # Marcadores para cortar a área de itens (no seu cupom tem esses termos) [file:comp4.jpg]
    ITEMS_HEADER_HINTS = [
        "SQ.CODIGO",
        "SQ. CODIGO",
        "DESCRICAO",
        "DESCRIÇÃO",
        "QTD",
        "VL.UNIT",
        "VL. UNIT",
        "TOTAL",
    ]
    STOP_HINTS = [
        "QTD. TOTAL DE ITENS",
        "QTD TOTAL DE ITENS",
        "VALOR TOTAL",
        "CARTAO",
        "CARTÃO",
        "CONSUMIDOR",
        "CONSULTE PELA CHAVE",
    ]

    def __init__(self, use_gpu: bool = False):
        self.debug_log: List[str] = []
        try:
            params = {
                "use_angle_cls": True,   # melhora em fotos tortas de cupom [file:comp4.jpg]
                "lang": "pt",
                "show_log": False,
            }
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
                ("clahe", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)),
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
                self._log("❌ IMG IS NONE")
                return []

            result = self.ocr.ocr(img)
            if result is None:
                self._log("❌ RESULT IS NONE")
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

                for item in raw_lines or []:
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

            # ordena por posição vertical
            lines.sort(key=lambda x: x["y_position"])
            return lines

        except Exception as e:
            self._log(f"❌ ERRO extract_text: {str(e)}")
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

        # recorta para área provável de itens e remove rodapé [file:comp4.jpg]
        item_lines = self._slice_item_area(ocr_lines)

        itens = self._extract_items_by_line_regex(item_lines, tipo)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]["data"] if qr_data else None,
            "mensagem": None if itens else "Nenhum item detectado",
            "confianca": 1.0 if itens else 0.0,
        }

    # ---------------- SLICE: ÁREA DE ITENS ----------------
    def _slice_item_area(self, ocr_lines: List[Dict]) -> List[Dict]:
        texts = [l.get("text", "") for l in ocr_lines]
        upper = [t.upper().strip() for t in texts]

        # encontra uma linha que pareça cabeçalho da tabela (contém DESCRICAO e QTD etc.) [file:comp4.jpg]
        start_idx = 0
        for i, t in enumerate(upper):
            if ("DESCR" in t) and ("QTD" in t or "QT" in t) and ("TOTAL" in t or "VL" in t):
                start_idx = i + 1
                break

        # encontra rodapé por hints [file:comp4.jpg]
        end_idx = len(texts)
        for i in range(start_idx, len(upper)):
            if any(h in upper[i] for h in self.STOP_HINTS):
                end_idx = i
                break

        sliced = ocr_lines[start_idx:end_idx]

        # fallback extra: se ainda vier muita coisa, corta ao ver "QTD. TOTAL DE ITENS" etc.
        cleaned: List[Dict] = []
        for l in sliced:
            t = (l.get("text", "") or "").upper()
            if "QTD. TOTAL DE ITENS" in t or "VALOR TOTAL" in t:
                break
            cleaned.append(l)

        return cleaned

    # ---------------- PARSER: POR LINHA (REGEX) ----------------
    def _extract_items_by_line_regex(self, lines: List[Dict], tipo: str) -> List[Dict]:
        """
        Para cupons como o anexado, cada item geralmente cabe em uma única linha: [file:comp4.jpg]
          NN CODIGO DESCRICAO ... QTD UN X VL_UNIT ST TOTAL

        Estratégia:
        - Filtrar linhas que começam com NN + código (8-14 dígitos).
        - Extrair:
          - descrição (lazy) até achar a quantidade+unidade
          - quantidade + unidade (UN/KG/LT/...)
          - unitário após X (opcional)
          - total = último preço na linha
        - Se unitário não existir, calcula total/qtd.
        """
        data_compra = self._extract_date("\n".join((l.get("text", "") or "") for l in lines))

        # Normaliza pequenos ruídos comuns de OCR
        def norm(s: str) -> str:
            s = s.replace("×", "X")
            s = re.sub(r"\s+", " ", s).strip()
            return s

        itens: List[Dict] = []

        # Ex.: "01 07891515546335 BATATA SADIA 1,05KG 1 UN X 15,89 T03 15,89" [file:comp4.jpg]
        line_pattern = re.compile(
            r"^\s*(?P<nn>\d{1,2})\s+(?P<code>\d{8,14})\s+"
            r"(?P<body>.+?)\s*$"
        )

        # Dentro do body, localizar o trecho de quantidade/unidade e unitário opcional:
        # aceita "1 UN X 15,89" ou "0,546 KG X 26,90" etc. [file:comp4.jpg]
        qtd_pattern = re.compile(
            r"(?P<desc>.+?)\s+"
            r"(?P<qtd>\d+[.,]?\d*)\s*(?P<un>UN|KG|LT|L|PC|PCT|CX|FD)\b"
            r"(?:\s*[xX]\s*(?P<unit>\d+[.,]\d{2}))?",
            flags=re.IGNORECASE
        )

        money_pattern = re.compile(r"\d+[.,]\d{2}")

        for l in lines:
            raw = l.get("text", "") or ""
            raw = norm(raw)
            if not raw:
                continue

            m = line_pattern.match(raw)
            if not m:
                continue

            body = m.group("body")

            # total = último dinheiro na linha (mais estável pro layout do cupom) [file:comp4.jpg]
            monies = money_pattern.findall(raw)
            if not monies:
                continue
            valor_total = self._to_float(monies[-1])

            qm = qtd_pattern.search(body)
            if not qm:
                # se OCR quebrou e não achou qtd, ignora (melhor do que inventar)
                continue

            desc_raw = qm.group("desc")
            quantidade = self._to_float(qm.group("qtd")) or 1.0

            unit_raw = qm.group("unit")
            valor_unitario = self._to_float(unit_raw) if unit_raw else None

            desc = self._clean_desc(desc_raw)

            # se unitário não veio, calcula por total/qtd
            if valor_unitario is None and valor_total is not None and quantidade > 0:
                valor_unitario = round(valor_total / quantidade, 2)

            itens.append(
                {
                    "item": desc,
                    "quantidade": quantidade,
                    "valor_unitario": valor_unitario,
                    "valor_total": valor_total,
                    "data_compra": data_compra if tipo == "gasto" else None,
                    "data_venda": data_compra if tipo == "venda" else None,
                }
            )

        return itens

    # ---------------- HELPERS ----------------
    def _to_float(self, s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        try:
            return float(s.replace(".", "").replace(",", ".") if s.count(",") == 1 and s.count(".") > 1 else s.replace(",", "."))
        except Exception:
            try:
                return float(s.replace(",", "."))
            except Exception:
                return None

    def _clean_desc(self, desc: str) -> str:
        if not desc:
            return "ITEM DESCONHECIDO"

        # remove lixo inicial (se algum dia entrar NN etc.)
        desc = re.sub(r"^\d+\s*", "", desc)

        # remove sufixos de unidade soltos
        desc = re.sub(r"\s+(UN|KG|LT|PC|L|M)\s*$", "", desc, flags=re.IGNORECASE)

        # normaliza espaços e upper
        desc = re.sub(r"\s+", " ", desc).strip().upper()

        # remove caracteres estranhos
        desc = re.sub(r"[^A-Z0-9À-Ü\s\-.,/]", "", desc)

        if desc in {"UN", "X", "F", "-", ""}:
            return "ITEM DESCONHECIDO"

        for wrong, right in self.COMMON_CORRECTIONS.items():
            if wrong in desc:
                desc = desc.replace(wrong, right)

        return desc if desc else "ITEM DESCONHECIDO"

    # ---------------- DATA ----------------
    def _extract_date(self, text: str) -> str:
        patterns = [
            r"emiss[aã]o[:\s]*(\d{2}/\d{2}/\d{4})",
            r"(\d{2}/\d{2}/\d{4})",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return datetime.now().strftime("%d/%m/%Y")