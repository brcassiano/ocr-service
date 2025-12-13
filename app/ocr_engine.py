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

    # cabeçalho de item: "01 07891515546335BATATA ..."
    RE_ITEM_HEADER = re.compile(r"^\s*(?P<sq>\d{2})\s+(?P<code>\d{8,14})\s*(?P<desc>.+)$")

    # pega "1UNx15,89" | "0,546KGx26,90" | "1 UH x 14,99" | variações com espaços
    RE_QTD_UNIT_X_VLUNIT = re.compile(
        r"(?P<qtd>\d+(?:[.,]\d+)?)\s*(?P<un>[A-Z]{1,3})\s*[xX]\s*(?P<vl>\d+(?:[.,]\d{2}))",
        re.IGNORECASE,
    )

    # dinheiro BR: 15,89 / 236,09 etc
    RE_MONEY = re.compile(r"\d+(?:[.,]\d{2})")

    COMMON_CORRECTIONS = {
        "ZER0": "ZERO",
        "I0G": "IOG",
        "OUOS": "OVOS",
        "PA0": "PAO",
        "P.QUEI.JO": "P.QUEIJO",
        "P.QUEIJ0": "P.QUEIJO",
        "UOS": "OVOS",
        "UH": "UN",
        "UL.UNIT": "VL.UNIT",
        "UL.UNIT": "VL.UNIT",
    }

    def __init__(self, use_gpu: bool = False):
        params = {
            "use_angle_cls": True,
            "lang": "pt",
            "show_log": False,
        }
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

        itens = self._extract_items_nfce_sp_layout(ocr_lines, tipo, full_text)

        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_data[0]["data"] if qr_data else None,
            "mensagem": None if itens else "Nenhum item detectado",
            "confianca": 1.0 if itens else 0.0,
        }

    # ----------- parser determinístico NFC-e SP (por colunas) -----------
    def _extract_items_nfce_sp_layout(self, ocr_lines: List[Dict], tipo: str, full_text: str) -> List[Dict]:
        """
        Regras:
        - Cada item começa com "SQ CODIGO ..." (SQ 2 dígitos).
        - Na mesma linha visual (mesmo y) aparecem tokens adicionais: QTD x VLUNIT (com UN/KG) e TOTAL (às vezes separado).
        - Para cada item: pega sq, descricao, qtd, vl_unit e total.
        """
        data_compra = self._extract_date(full_text)

        # 1) recorta área de itens: depois do cabeçalho "SQ.CODIGO"
        start_y = None
        end_y = None
        for t in ocr_lines:
            up = (t.get("text") or "").upper()
            if "SQ.CODIGO" in up or "SQ. CODIGO" in up:
                start_y = t.get("y_position", 0)
                break

        if start_y is None:
            start_y = 0

        for t in ocr_lines:
            up = (t.get("text") or "").upper()
            if any(h in up for h in self.STOP_HINTS):
                end_y = t.get("y_position", 10**9)
                break

        if end_y is None:
            end_y = 10**9

        tokens = [t for t in ocr_lines if start_y + 5 <= t["y_position"] <= end_y - 5]

        # 2) agrupa por linha visual usando y tolerance
        grouped = self._group_by_y(tokens, y_tol=10)

        itens: List[Dict] = []
        current = None

        def flush():
            nonlocal current
            if not current:
                return
            # valida mínimos
            if current.get("descricao") and current.get("valor_total") is not None:
                itens.append(
                    {
                        "item": current["descricao"],
                        "quantidade": current.get("quantidade"),
                        "valor_unitario": current.get("valor_unitario"),
                        "valor_total": current.get("valor_total"),
                        "data_compra": data_compra if tipo == "gasto" else None,
                        "data_venda": data_compra if tipo == "venda" else None,
                    }
                )
            current = None

        for line in grouped:
            # line: tokens da mesma linha visual, ordenados por x
            text_join = " ".join([tok["text"] for tok in line]).strip()
            text_join = self._norm_text(text_join)

            # ignora cabeçalhos
            up = text_join.upper()
            if "DESCRICAO" in up and "TOTAL" in up:
                continue

            m = self.RE_ITEM_HEADER.match(text_join)
            if m:
                # novo item: flush o anterior
                flush()

                sq = m.group("sq")
                desc_part = m.group("desc")

                # desc_part pode vir grudado com o código (sem espaço); já está no grupo desc
                desc = self._clean_desc(desc_part)

                current = {
                    "sq": sq,
                    "descricao": desc,
                    "quantidade": None,
                    "valor_unitario": None,
                    "valor_total": None,
                }

                # tenta achar total na mesma linha (às vezes aparece no final)
                monies = self.RE_MONEY.findall(text_join)
                if monies:
                    current["valor_total"] = self._to_float(monies[-1])

                # tenta achar qtd/vl_unit na mesma linha (às vezes vem grudado tipo "...0,546KGx26,90T03")
                q = self.RE_QTD_UNIT_X_VLUNIT.search(text_join)
                if q:
                    current["quantidade"] = self._to_float(q.group("qtd"))
                    current["valor_unitario"] = self._to_float(q.group("vl"))

                continue

            # linhas complementares: só processa se já existe item aberto
            if not current:
                continue

            # 1) total isolado (token tipo "15,89")
            monies = self.RE_MONEY.findall(text_join)
            if monies:
                # regra: se só tem 1 valor monetário e current.total ainda vazio, assume total
                if len(monies) == 1 and current.get("valor_total") is None:
                    current["valor_total"] = self._to_float(monies[0])
                # se tem mais de um, pega o último como total (padrão NFC-e)
                elif len(monies) >= 2:
                    current["valor_total"] = self._to_float(monies[-1])

            # 2) qtd e vl_unit (token tipo "1UNx15,89T03" ou "1UN x28,90 T04")
            q = self.RE_QTD_UNIT_X_VLUNIT.search(text_join)
            if q:
                current["quantidade"] = self._to_float(q.group("qtd"))
                current["valor_unitario"] = self._to_float(q.group("vl"))

            # Heurística: quando já tem total e (qtd ou vl_unit), pode fechar item ao encontrar próxima linha
            # (não flush aqui; flush ocorre ao detectar próximo header ou no final)

        flush()
        return itens

    def _group_by_y(self, tokens: List[Dict], y_tol: int = 10) -> List[List[Dict]]:
        # ordena por y e agrupa por proximidade
        toks = sorted(tokens, key=lambda t: (t["y_position"], t["x_position"] if t["x_position"] is not None else 10**9))
        groups: List[List[Dict]] = []
        refs: List[int] = []

        for t in toks:
            y = int(t["y_position"])
            placed = False
            for idx, yref in enumerate(refs):
                if abs(y - yref) <= y_tol:
                    groups[idx].append(t)
                    # atualiza referência (média simples)
                    refs[idx] = int((refs[idx] + y) / 2)
                    placed = True
                    break
            if not placed:
                groups.append([t])
                refs.append(y)

        # ordena cada grupo por x
        for g in groups:
            g.sort(key=lambda t: t["x_position"] if t["x_position"] is not None else 10**9)

        # ordena grupos por yref
        groups = [g for _, g in sorted(zip(refs, groups), key=lambda x: x[0])]
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
            # normaliza "1.234,56" e "1234,56" e "1234.56"
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