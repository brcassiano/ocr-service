from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode

import cv2
import numpy as np
import re

from datetime import datetime
from typing import List, Dict, Optional, Tuple

import logging

from .utils import TextProcessor

logger = logging.getLogger(__name__)


class OCREngine:
    KEYWORDS_VENDA = ["recebido", "pix recebido", "crédito em conta", "depósito", "recibo"]

    COMMON_CORRECTIONS = {
        "ALHOTRADIC": "ALHO TRADIC",
        "QJ": "QUEIJO",
        "ZER0": "ZERO",
        "I0G": "IOG",
        "OUOS": "OVOS",
        "PA0": "PAO",
        "P.QUEI.JO": "P.QUEIJO",
    }

    STOP_HINTS = [
        "QTD. TOTAL DE ITENS",
        "QTD TOTAL DE ITENS",
        "VALOR TOTAL",
        "CARTAO",
        "CARTÃO",
        "CONSUMIDOR",
        "CONSULTE PELA CHAVE",
        "CHAVE DE ACESSO",
        "PROTOCOLO",
    ]

    def __init__(self, use_gpu: bool = False):
        self.debug_log: List[str] = []
        try:
            params = {"use_angle_cls": True, "lang": "pt", "show_log": False}
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
                            return [{"data": obj.data.decode("utf-8", errors="ignore"), "type": obj.type}]

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
                        if poly and isinstance(poly[0], (list, tuple)):
                            y_pos = int(poly[0][1])

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
                                "confidence": round(float(confidence), 3),
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

        # 1) recorta área de itens (após o header e antes do rodapé)
        item_tokens = self._slice_item_area(ocr_lines)

        # 2) reconstrói "linhas" juntando tokens por y
        reconstructed = self._reconstruct_lines_by_y(item_tokens, y_tol=10)

        # 3) extrai itens (agora sim, com linhas completas)
        itens = self._extract_items_from_reconstructed_lines(reconstructed, tipo)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]["data"] if qr_data else None,
            "mensagem": None if itens else "Nenhum item detectado",
            "confianca": 1.0 if itens else 0.0,
        }

    # ---------------- SLICE: ÁREA DE ITENS ----------------
    def _slice_item_area(self, ocr_lines: List[Dict]) -> List[Dict]:
        # acha o cabeçalho: no seu raw aparece SQ.CODIGO / DESCRICAO / QTD / VL.UNIT / ST / TOTAL [conversation_history:query]
        start_idx = 0
        for i, l in enumerate(ocr_lines):
            t = (l.get("text") or "").upper()
            if "SQ.CODIGO" in t or "SQ. CODIGO" in t:
                start_idx = i + 1
                break

        # corta no rodapé [conversation_history:query]
        end_idx = len(ocr_lines)
        for i in range(start_idx, len(ocr_lines)):
            t = (ocr_lines[i].get("text") or "").upper()
            if any(h in t for h in self.STOP_HINTS):
                end_idx = i
                break

        return ocr_lines[start_idx:end_idx]

    # ---------------- RECONSTRUIR LINHAS POR Y ----------------
    def _reconstruct_lines_by_y(self, tokens: List[Dict], y_tol: int = 10) -> List[str]:
        """
        PaddleOCR frequentemente devolve cada coluna como um token separado; aqui juntamos por proximidade de y. [conversation_history:query]
        """
        # normaliza
        cleaned = []
        for t in tokens:
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            cleaned.append(
                {
                    "text": txt.replace("×", "X"),
                    "y": int(t.get("y_position") or 0),
                }
            )

        # agrupa em bandas
        groups: List[Tuple[int, List[str]]] = []  # (y_ref, [texts])
        for tok in cleaned:
            placed = False
            for gi in range(len(groups)):
                y_ref, arr = groups[gi]
                if abs(tok["y"] - y_ref) <= y_tol:
                    arr.append(tok["text"])
                    # ajusta y_ref para média simples (estabiliza)
                    new_y = int((y_ref + tok["y"]) / 2)
                    groups[gi] = (new_y, arr)
                    placed = True
                    break
            if not placed:
                groups.append((tok["y"], [tok["text"]]))

        # ordena por y e junta textos na ordem em que chegaram (boa o suficiente pro seu output)
        groups.sort(key=lambda x: x[0])

        lines = []
        for _, parts in groups:
            line = " ".join(parts)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)
        return lines

    # ---------------- EXTRATOR DE ITENS (COM MERGE DE LINHAS QUEBRADAS) ----------------
    def _extract_items_from_reconstructed_lines(self, lines: List[str], tipo: str) -> List[Dict]:
        data_compra = self._extract_date("\n".join(lines))

        header_pat = re.compile(r"^\s*(\d{1,2})\s+(\d{8,14})\b")
        money_pat = re.compile(r"\d+[.,]\d{2}")

        # captura qty/un, unit e total ao final (último dinheiro da linha)
        qty_pat = re.compile(r"(?P<qtd>\d+[.,]?\d*)\s*(?P<un>UN|KG|LT|L|PC|PCT|CX|FD)\b", re.IGNORECASE)
        unit_pat = re.compile(r"\b[Xx]\s*(?P<unit>\d+[.,]\d{2})")

        itens: List[Dict] = []

        i = 0
        while i < len(lines):
            line = self._norm(lines[i])

            # só começa item se tiver NN + código
            if not header_pat.search(line):
                i += 1
                continue

            # merge: junta próximas linhas até encontrar UN/KG e pelo menos 1 valor monetário
            merged = line
            j = i + 1
            while j < len(lines):
                if header_pat.search(self._norm(lines[j])):  # próximo item começou
                    break

                merged_try = self._norm(merged + " " + lines[j])
                # critério: se já temos quantidade+un e algum dinheiro, pode parar
                if qty_pat.search(merged_try) and money_pat.search(merged_try):
                    merged = merged_try
                    j += 1
                    break

                merged = merged_try
                j += 1

            # agora parseia merged
            # remove "NN codigo" do começo para sobrar corpo
            body = header_pat.sub("", merged).strip()

            # total = último dinheiro encontrado
            monies = money_pat.findall(merged)
            if not monies:
                i = j
                continue
            valor_total = self._to_float(monies[-1])

            qm = qty_pat.search(body)
            if not qm:
                i = j
                continue

            quantidade = self._to_float(qm.group("qtd")) or 1.0

            um = unit_pat.search(body)
            valor_unitario = self._to_float(um.group("unit")) if um else None
            if valor_unitario is None and valor_total is not None and quantidade > 0:
                valor_unitario = round(valor_total / quantidade, 2)

            # descrição: tudo antes do match de quantidade/un
            desc_raw = body[: qm.start()].strip()
            desc = self._clean_desc(desc_raw)

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

            i = j

        return itens

    # ---------------- HELPERS ----------------
    def _norm(self, s: str) -> str:
        s = (s or "").replace("×", "X")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _to_float(self, s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s = s.strip()
        try:
            # padrão BR: 1.234,56 / 123,45 / 10.39 (OCR às vezes troca)
            s = s.replace(" ", "")
            if s.count(",") == 1 and s.count(".") >= 1:
                # assume pontos como milhar
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", ".")
            return float(s)
        except Exception:
            return None

    def _clean_desc(self, desc: str) -> str:
        if not desc:
            return "ITEM DESCONHECIDO"

        desc = desc.upper()
        desc = re.sub(r"\s+", " ", desc).strip()

        # FIX: hífen no final do charset para não virar range inválido
        desc = re.sub(r"[^A-Z0-9À-Ü\s\\.,/-]", "", desc)

        for wrong, right in self.COMMON_CORRECTIONS.items():
            if wrong in desc:
                desc = desc.replace(wrong, right)

        if desc in {"UN", "X", "F", "-", ""}:
            return "ITEM DESCONHECIDO"

        return desc


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