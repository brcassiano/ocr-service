import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode

logger = logging.getLogger(__name__)


class OCREngine:
    KEYWORDS_VENDA = ["recebido", "pix recebido", "crédito em conta", "depósito", "recibo"]

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

    COMMON_CORRECTIONS = {
        "ZER0": "ZERO",
        "I0G": "IOG",
        "OUOS": "OVOS",
        "PA0": "PAO",
        "P.QUEI.JO": "P.QUEIJO",
    }

    def __init__(self, use_gpu: bool = False):
        # PaddleOCR 3.x: use_angle_cls na init, não no call
        params = {"use_angle_cls": True, "lang": "pt"}
        if use_gpu:
            params["use_gpu"] = True
        self.ocr = PaddleOCR(**params)

    # ---------------- QR CODE ----------------
    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for processed in [
                gray,
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray),
                cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
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

    # ---------------- OCR: tentativas progressivas ----------------
    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        """
        Retorna tokens OCR com text/confidence/x_position/y_position.
        Compatível com PaddleOCR 3.x (sem cls=True no call). [web:38]
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("cv2.imdecode retornou None")
                return []

            # tentativa 1: imagem original
            result = self.ocr.ocr(img)
            lines = self._normalize_ocr_result_to_lines(result)
            if lines:
                logger.info(f"OCR sucesso (raw): {len(lines)} linhas")
                lines.sort(key=lambda x: (x["y_position"], x["x_position"] if x["x_position"] is not None else 10**9))
                return lines

            # tentativa 2: pré-processamento básico (binarização)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result2 = self.ocr.ocr(thr)
            lines = self._normalize_ocr_result_to_lines(result2)
            if lines:
                logger.info(f"OCR sucesso (thresh): {len(lines)} linhas")
                lines.sort(key=lambda x: (x["y_position"], x["x_position"] if x["x_position"] is not None else 10**9))
                return lines

            # tentativa 3: redimensionar (para imagens muito pequenas)
            h, w = img.shape[:2]
            if h < 600 or w < 400:
                scale = max(800 / w, 1200 / h)
                resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                result3 = self.ocr.ocr(resized)
                lines = self._normalize_ocr_result_to_lines(result3)
                if lines:
                    logger.info(f"OCR sucesso (resize): {len(lines)} linhas")
                    lines.sort(key=lambda x: (x["y_position"], x["x_position"] if x["x_position"] is not None else 10**9))
                    return lines

            logger.warning("Nenhuma linha OCR encontrada após 3 tentativas")
            return []

        except Exception as e:
            logger.error(f"extract_text error: {e}", exc_info=True)
            return []

    def _normalize_ocr_result_to_lines(self, result: Any) -> List[Dict]:
        """
        Aceita diferentes formatos e extrai lista de:
        {text, confidence, x_position, y_position}
        """
        if result is None:
            return []

        # Em PaddleOCR 3.x, formato comum: [ [ [poly, (text,score)], ... ] ]
        # ou: [ [item, item, ...] ]
        if isinstance(result, list) and result:
            # às vezes vem: [page0] ou já vem direto lista
            candidates = result[0] if isinstance(result[0], list) else result

            out: List[Dict] = []
            for item in candidates or []:
                parsed = self._parse_single_ocr_item(item)
                if parsed:
                    out.append(parsed)
            return out

        return []

    def _parse_single_ocr_item(self, item: Any) -> Optional[Dict]:
        """
        Espera algo como:
          - [poly, (text, score)]
          - (poly, (text, score))
          - dict-like (algumas pipelines)
        """
        poly = None
        text = None
        conf = None

        # formato clássico list/tuple
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            poly = item[0]
            rec = item[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text = rec[0]
                conf = rec[1]

        # formato dict (defensivo)
        if isinstance(item, dict):
            poly = item.get("poly") or item.get("points") or item.get("box")
            text = item.get("text")
            conf = item.get("score") or item.get("confidence")

        if text is None:
            return None

        try:
            text = str(text).strip().replace("×", "X")
        except Exception:
            return None

        try:
            conf = float(conf) if conf is not None else 0.0
        except Exception:
            conf = 0.0

        if not text or conf < 0.4:
            return None

        x_pos, y_pos = self._xy_from_poly(poly)

        return {
            "text": text,
            "confidence": round(conf, 3),
            "y_position": int(y_pos),
            "x_position": int(x_pos) if x_pos is not None else None,
        }

    def _xy_from_poly(self, poly: Any) -> Tuple[Optional[int], int]:
        """
        poly esperado: [[x,y],...]
        """
        try:
            if poly is None:
                return None, 0
            xs = []
            ys = []
            for p in poly:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    xs.append(int(p[0]))
                    ys.append(int(p[1]))
            if not xs or not ys:
                return None, 0
            return min(xs), min(ys)
        except Exception:
            return None, 0

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

        item_tokens = self._slice_item_area(ocr_lines)
        reconstructed = self._reconstruct_lines_by_yx(item_tokens, y_tol=12)
        itens = self._extract_items_from_lines(reconstructed, tipo, full_text)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]["data"] if qr_data else None,
            "mensagem": None if itens else "Nenhum item detectado",
            "confianca": 1.0 if itens else 0.0,
        }

    def _slice_item_area(self, ocr_lines: List[Dict]) -> List[Dict]:
        start_idx = 0
        for i, l in enumerate(ocr_lines):
            t = (l.get("text") or "").upper()
            if "SQ.CODIGO" in t or "SQ. CODIGO" in t:
                start_idx = i + 1
                break

        end_idx = len(ocr_lines)
        for i in range(start_idx, len(ocr_lines)):
            t = (ocr_lines[i].get("text") or "").upper()
            if any(h in t for h in self.STOP_HINTS):
                end_idx = i
                break

        return ocr_lines[start_idx:end_idx]

    def _reconstruct_lines_by_yx(self, tokens: List[Dict], y_tol: int = 12) -> List[str]:
        cleaned = []
        for t in tokens:
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            cleaned.append(
                {
                    "text": txt,
                    "y": int(t.get("y_position") or 0),
                    "x": int(t["x_position"]) if t.get("x_position") is not None else 10**9,
                }
            )

        groups: List[Dict] = []
        for tok in cleaned:
            placed = False
            for g in groups:
                if abs(tok["y"] - g["y_ref"]) <= y_tol:
                    g["items"].append(tok)
                    g["y_ref"] = int((g["y_ref"] + tok["y"]) / 2)
                    placed = True
                    break
            if not placed:
                groups.append({"y_ref": tok["y"], "items": [tok]})

        groups.sort(key=lambda g: g["y_ref"])

        lines: List[str] = []
        for g in groups:
            g["items"].sort(key=lambda it: it["x"])
            line = " ".join([it["text"] for it in g["items"]])
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)

        return lines

    def _extract_items_from_lines(self, lines: List[str], tipo: str, full_text: str) -> List[Dict]:
        data_compra = self._extract_date(full_text)

        header_pat = re.compile(r"^\s*(?P<n>\d{1,2})\s+(?P<code>\d{8,14})\b")
        money_pat = re.compile(r"\d+[.,]\d{2}")
        qty_pat = re.compile(r"(?P<qtd>\d+[.,]?\d*)\s*(?P<un>UN|KG|LT|L|PC|PCT|CX|FD)\b", re.IGNORECASE)
        unit_pat = re.compile(r"\b[Xx]\s*(?P<unit>\d+[.,]\d{2})")

        itens: List[Dict] = []
        i = 0

        while i < len(lines):
            line = self._norm(lines[i])
            if not header_pat.search(line):
                i += 1
                continue

            merged = line
            j = i + 1
            while j < len(lines):
                nxt = self._norm(lines[j])
                if header_pat.search(nxt):
                    break
                merged_try = self._norm(merged + " " + nxt)
                if qty_pat.search(merged_try) and money_pat.search(merged_try):
                    merged = merged_try
                    j += 1
                    break
                merged = merged_try
                j += 1

            body = header_pat.sub("", merged).strip()

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

    def _norm(self, s: str) -> str:
        s = (s or "").replace("×", "X")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _to_float(self, s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s = s.strip().replace(" ", "")
        try:
            if s.count(",") == 1 and s.count(".") >= 1:
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
        desc = re.sub(r"[^A-Z0-9À-Ü\s.,/-]", "", desc)
        for wrong, right in self.COMMON_CORRECTIONS.items():
            desc = desc.replace(wrong, right)
        desc = desc.strip(" -")
        return desc if desc else "ITEM DESCONHECIDO"

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