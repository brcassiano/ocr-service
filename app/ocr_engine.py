import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

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

    RE_ITEM_HEADER = re.compile(r"^\s*(?P<sq>\d{2})\s+(?P<code>\d{8,14})(?P<desc>.*)$")

    # qtd/un/valor: funciona tanto no bloco do meio quanto quando vem grudado no bloco esquerdo
    RE_QTD_X_UNIT = re.compile(
        r"(?P<qtd>\d+(?:[.,]\d+)?)\s*(?P<un>[A-Z]{1,3})\s*[xX]\s*(?P<vl>\d+(?:[.,]\d{2}))",
        re.IGNORECASE,
    )

    RE_MONEY = re.compile(r"\d+(?:[.,]\d{2})")

    COMMON_CORRECTIONS = {
        "ZER0": "ZERO",
        "I0G": "IOG",
        "OUOS": "OVOS",
        "UOS": "OVOS",
        "UH": "UN",
        "1Ux": "1UNx",
        "SUIFT": "SWIFT",
    }

    def __init__(self, use_gpu: bool = False):
        params = {"use_angle_cls": True, "lang": "pt", "show_log": False}
        if use_gpu:
            params["use_gpu"] = True
        self.ocr = PaddleOCR(**params)

    # ---------------- QR CODE ----------------
    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        try:
            img = self._decode_image_bgr(image_bytes)
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

    # ---------------- OCR ----------------
    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        img = self._decode_image_bgr(image_bytes)
        if img is None:
            logger.warning("Imagem inválida (decode retornou None)")
            return []

        attempts = [
            ("raw", img),
            ("thresh_bgr", self._to_bgr(self._threshold(img))),
            ("zoom", cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)),
        ]

        for name, attempt_img in attempts:
            try:
                result = self.ocr.ocr(attempt_img, cls=True)
                lines = self._normalize_paddle2_result(result)
                if lines:
                    lines.sort(key=lambda x: (x["y_position"], x["x_position"] if x["x_position"] is not None else 10**9))
                    logger.info(f"OCR sucesso ({name}): {len(lines)} tokens")
                    return lines
                logger.info(f"OCR vazio ({name})")
            except Exception as e:
                logger.error(f"OCR erro ({name}): {e}", exc_info=True)

        return []

    def _normalize_paddle2_result(self, result) -> List[Dict]:
        if not result:
            return []
        page = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
        if not page:
            return []

        out: List[Dict] = []
        for item in page:
            try:
                box = item[0]
                rec = item[1]
                text = str(rec[0]).strip()
                conf = float(rec[1])
                if not text or conf < 0.35:
                    continue
                x_pos, y_pos = self._xy_from_box(box)
                out.append(
                    {
                        "text": self._norm_text(text),
                        "confidence": round(conf, 3),
                        "y_position": int(y_pos),
                        "x_position": int(x_pos) if x_pos is not None else None,
                    }
                )
            except Exception:
                continue
        return out

    def _xy_from_box(self, box) -> Tuple[Optional[int], int]:
        try:
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
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

        itens = self._extract_items_nfce_sp_by_columns(ocr_lines, tipo, full_text)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]["data"] if qr_data else None,
            "mensagem": None if itens else "Nenhum item detectado",
            "confianca": 1.0 if itens else 0.0,
        }

    def _extract_items_nfce_sp_by_columns(self, ocr_lines: List[Dict], tipo: str, full_text: str) -> List[Dict]:
        data_compra = self._extract_date(full_text)

        # faixa de itens
        start_y = 0
        for t in ocr_lines:
            up = (t.get("text") or "").upper()
            if "SQ.CODIGO" in up or "SQ. CODIGO" in up:
                start_y = t.get("y_position", 0) + 5
                break

        end_y = 10**9
        for t in ocr_lines:
            up = (t.get("text") or "").upper()
            if any(h in up for h in self.STOP_HINTS):
                end_y = t.get("y_position", 10**9) - 5
                break

        tokens = [t for t in ocr_lines if start_y <= t["y_position"] <= end_y]

        # colunas (do seu exemplo)
        X_LEFT_MAX = 620
        X_MID_MIN = 620
        X_RIGHT_MIN = 900

        groups = self._group_by_y(tokens, y_tol=6)

        def next_group(i: int, max_delta: int = 18) -> Optional[Dict]:
            if i + 1 >= len(groups):
                return None
            if groups[i + 1]["y_ref"] - groups[i]["y_ref"] <= max_delta:
                return groups[i + 1]
            return None

        itens_by_sq: Dict[str, Dict] = {}

        for i, g in enumerate(groups):
            left_tokens = [t for t in g["tokens"] if (t["x_position"] or 0) < X_LEFT_MAX]
            if not left_tokens:
                continue

            left_text = self._norm_text(" ".join([t["text"] for t in left_tokens]))
            m = self.RE_ITEM_HEADER.match(left_text)
            if not m:
                continue

            sq = m.group("sq")
            desc_raw = m.group("desc") or ""
            desc = self._clean_desc(desc_raw)

            # 1) tentar qtd/vl_unit do meio
            mid_tokens = [t for t in g["tokens"] if X_MID_MIN <= (t["x_position"] or 0) < X_RIGHT_MIN]
            mid_text = self._norm_text(" ".join([t["text"] for t in mid_tokens]))
            q = self.RE_QTD_X_UNIT.search(mid_text)

            # 2) fallback: às vezes vem grudado no bloco esquerdo (ex.: "0,546KGx26,90T03")
            if not q:
                q = self.RE_QTD_X_UNIT.search(left_text)

            quantidade = self._to_float(q.group("qtd")) if q else None
            valor_unitario = self._to_float(q.group("vl")) if q else None

            # 3) total: coluna direita da mesma linha -> senão coluna direita da linha de baixo
            valor_total = None
            right_tokens = [t for t in g["tokens"] if (t["x_position"] or 0) >= X_RIGHT_MIN]
            valor_total = self._parse_total_from_tokens(right_tokens)

            if valor_total is None:
                g2 = next_group(i, max_delta=18)
                if g2:
                    right_tokens_2 = [t for t in g2["tokens"] if (t["x_position"] or 0) >= X_RIGHT_MIN]
                    valor_total = self._parse_total_from_tokens(right_tokens_2)

            # 4) fallback final: se não achou total, mas qtd==1 e tem vl_unit => total = vl_unit
            if valor_total is None and valor_unitario is not None:
                if quantidade is None or abs(quantidade - 1.0) < 1e-6:
                    valor_total = valor_unitario

            # filtros mínimos
            if not desc or desc == "ITEM DESCONHECIDO":
                continue
            if valor_total is None:
                continue

            if quantidade is None:
                quantidade = 1.0
            if valor_unitario is None and quantidade and quantidade > 0:
                valor_unitario = round(valor_total / quantidade, 2)

            # dedupe por SQ: se já existe, mantém o que tem mais campos preenchidos
            candidate = {
                "item": desc,
                "quantidade": float(quantidade) if quantidade is not None else None,
                "valor_unitario": float(valor_unitario) if valor_unitario is not None else None,
                "valor_total": float(valor_total),
                "data_compra": data_compra if tipo == "gasto" else None,
                "data_venda": data_compra if tipo == "venda" else None,
                "_sq": sq,
            }

            prev = itens_by_sq.get(sq)
            if not prev:
                itens_by_sq[sq] = candidate
            else:
                prev_score = self._item_score(prev)
                cand_score = self._item_score(candidate)
                if cand_score >= prev_score:
                    itens_by_sq[sq] = candidate

        # ordenar por sq
        itens = list(itens_by_sq.values())
        try:
            itens.sort(key=lambda it: int(it["_sq"]))
        except Exception:
            pass
        for it in itens:
            it.pop("_sq", None)
        return itens

    def _parse_total_from_tokens(self, tokens: List[Dict]) -> Optional[float]:
        """
        Total é tipicamente 1 token tipo '15,89', mas às vezes vem OCR zoado tipo 66'9.
        Aqui normaliza alguns erros comuns e tenta converter.
        """
        if not tokens:
            return None

        raw = self._norm_text(" ".join([t["text"] for t in tokens]))

        # tenta normal
        monies = self.RE_MONEY.findall(raw)
        if monies:
            return self._to_float(monies[-1])

        # fallback OCR: 66'9 => 6,69 ou 66,9 (ambíguo). Para NFC-e, quase sempre 2 casas.
        # Regra: se tem 3 dígitos e um separador estranho, assume última é centavo: "669" => "6,69"
        cleaned = raw.replace("'", "").replace("`", "").replace(" ", "")
        cleaned = re.sub(r"[^0-9]", "", cleaned)
        if len(cleaned) == 3:
            guess = f"{cleaned[0]},{cleaned[1:]}"
            return self._to_float(guess)
        if len(cleaned) == 4:
            guess = f"{cleaned[:-2]},{cleaned[-2:]}"
            return self._to_float(guess)

        return None

    def _item_score(self, it: Dict) -> int:
        score = 0
        if it.get("item"):
            score += 1
        if it.get("quantidade") is not None:
            score += 1
        if it.get("valor_unitario") is not None:
            score += 1
        if it.get("valor_total") is not None:
            score += 1
        return score

    def _group_by_y(self, tokens: List[Dict], y_tol: int = 6) -> List[Dict]:
        toks = sorted(tokens, key=lambda t: (t["y_position"], t["x_position"] if t["x_position"] is not None else 10**9))
        groups: List[Dict] = []

        for t in toks:
            y = int(t["y_position"])
            placed = False
            for g in groups:
                if abs(y - g["y_ref"]) <= y_tol:
                    g["tokens"].append(t)
                    g["y_ref"] = int((g["y_ref"] + y) / 2)
                    placed = True
                    break
            if not placed:
                groups.append({"y_ref": y, "tokens": [t]})

        for g in groups:
            g["tokens"].sort(key=lambda t: t["x_position"] if t["x_position"] is not None else 10**9)

        groups.sort(key=lambda g: g["y_ref"])
        return groups

    # ---------------- Helpers ----------------
    def _decode_image_bgr(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _threshold(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thr

    def _to_bgr(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _norm_text(self, s: str) -> str:
        s = (s or "").replace("×", "X")
        s = re.sub(r"\s+", " ", s).strip()
        for wrong, right in self.COMMON_CORRECTIONS.items():
            s = s.replace(wrong, right)
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
        desc = (desc or "").upper()
        desc = re.sub(r"\s+", " ", desc).strip()
        desc = re.sub(r"[^A-Z0-9À-Ü\s\.,/-]", "", desc)
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