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

        # Remove rodapé a partir de "QTD. TOTAL DE ITENS" ou "VALOR TOTAL (R$)"
        cleaned_lines: List[Dict] = []
        stop = False
        for l in ocr_lines:
            t = l.get("text", "").upper()
            if "QTD. TOTAL DE ITENS" in t or "VALOR TOTAL (R$)" in t:
                stop = True
            if not stop:
                cleaned_lines.append(l)

        itens = self._extract_items_by_header_blocks(cleaned_lines, tipo)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]["data"] if qr_data else None,
            "mensagem": None,
            "confianca": 1.0 if itens else 0.0,
        }

    # ---------------- PARSER POR CABEÇALHO NN CÓDIGO ----------------
    def _extract_items_by_header_blocks(self, lines: List[Dict], tipo: str) -> List[Dict]:
        """
        Usa exclusivamente linhas que começam com 'NN CÓDIGO' como
        delimitadores de bloco de item. Cada bloco é [header_i, header_{i+1}).

        Dentro do bloco:
        - descrição: primeira linha com letras que não seja UN/X/F nem só preço;
        - quantidade: padrão '(\\d+[.,]?\\d*) UN' em alguma linha do bloco;
        - unitário: valor após 'x' na linha da quantidade;
        - total: preço na mesma linha da quantidade ou, em fallback,
                 maior preço dentro do bloco (sem rodapé).
        """
        data_compra = self._extract_date("\n".join(l.get("text", "") for l in lines))
        itens: List[Dict] = []

        # só texto, preservando ordem original
        texts = [l.get("text", "") for l in lines]

        # índices de cabeçalho
        header_indices: List[int] = []
        header_pattern = re.compile(r"^\s*(\d{1,2})\s+(\d{8,14})")

        for idx, text in enumerate(texts):
            if header_pattern.match(text):
                header_indices.append(idx)

        if not header_indices:
            return []

        header_indices.append(len(texts))  # sentinela final

        for h in range(len(header_indices) - 1):
            start = header_indices[h]
            end = header_indices[h + 1]
            block_lines = texts[start:end]
            if not block_lines:
                continue

            # remove o cabeçalho 'NN CODIGO' da primeira linha
            first = header_pattern.sub("", block_lines[0]).strip()
            block_rest = [first] + block_lines[1:]

            # descrição: primeira linha com letras que não seja UN/X/F e não seja só preço
            desc_raw = None
            for t in block_rest:
                t_stripped = t.strip()
                if not t_stripped:
                    continue
                if not re.search(r"[A-Za-zÀ-Ü]", t_stripped):
                    continue
                upper = t_stripped.upper()
                if upper in {"UN", "X", "F", "-"}:
                    continue
                if re.fullmatch(r"\d+[.,]\d{2}", t_stripped):
                    continue
                desc_raw = t_stripped
                break

            desc = self._clean_desc(desc_raw or "")

            # --------------- QUANTIDADE / UNITÁRIO / TOTAL ---------------
            quantidade = 1.0
            valor_unitario = None
            valor_total = None

            # linha de quantidade = primeira com " UN"
            qtd_line = None
            for t in block_rest:
                if re.search(r"\bUN\b", t, flags=re.IGNORECASE):
                    qtd_line = t
                    break

            if qtd_line:
                # quantidade
                qtd_match = re.search(r"(\d+[.,]?\d*)\s*UN", qtd_line, flags=re.IGNORECASE)
                if qtd_match:
                    try:
                        quantidade = float(qtd_match.group(1).replace(",", "."))
                    except ValueError:
                        quantidade = 1.0

                # preços nessa linha
                prices_in_qtd = [
                    float(p.replace(",", "."))
                    for p in re.findall(r"\d+[.,]\d{2}", qtd_line)
                ]

                # unitário após 'x'
                unit_match = re.search(r"[xX]\s*(\d+[.,]\d{2})", qtd_line)
                if unit_match:
                    try:
                        valor_unitario = float(unit_match.group(1).replace(",", "."))
                    except ValueError:
                        valor_unitario = None

                # se tiver 2 preços na linha e já temos unitário, assume primeiro como total
                if len(prices_in_qtd) >= 2 and valor_unitario is not None and valor_total is None:
                    valor_total = prices_in_qtd[0]
                elif len(prices_in_qtd) == 1 and valor_unitario is None:
                    # só um preço na linha, sem 'x': assume total
                    valor_total = prices_in_qtd[0]

            # fallback: se ainda não tiver total, procura preços nas linhas do bloco
            # (já sem rodapé, pois foi cortado em structure_data)
            if valor_total is None:
                safe_text = " ".join(block_rest)
                prices_block = [
                    float(p.replace(",", "."))
                    for p in re.findall(r"\d+[.,]\d{2}", safe_text)
                ]
                if prices_block:
                    # normalmente o maior preço do bloco é o total do item (ex: produtos a peso)
                    valor_total = max(prices_block)

            # se ainda não tiver unitário e tiver total + quantidade
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

    def _clean_desc(self, desc: str) -> str:
        if not desc:
            return "ITEM DESCONHECIDO"

        desc = re.sub(r"^\d+\s*", "", desc)
        desc = re.sub(r"\s+(UN|KG|LT|PC|L|M)\s*$", "", desc, flags=re.IGNORECASE)
        desc = re.sub(r"\s*[xX]\s*\d+[.,]\d{2}.*$", "", desc)
        desc = re.sub(r"\s*(T\d{2,3}|F)\s*\d+[.,]\d{2}.*$", "", desc)
        desc = re.sub(r"\s+", " ", desc).strip().upper()
        desc = re.sub(r"[^A-Z0-9À-Ü\s\-.,/]", "", desc)

        # se sobrou só UN/X/F/-, trata como desconhecido
        if desc in {"UN", "X", "F", "-"}:
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