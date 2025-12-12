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

    # ================ QR CODE ================
    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Tenta múltiplas técnicas de processamento
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

            # Fallback: usar detector nativo do OpenCV
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(img)
            if data:
                return [{"data": data, "type": "QRCODE"}]

            return None
        except Exception:
            return None

    # ================ OCR LINES ================
    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        self.debug_log = []
        self._log(f"=== extract_text INICIADO ===")
        try:
            self._log("1. Decodificando imagem...")
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                self._log("❌ IMG IS NONE")
                return []

            self._log(f"✅ Imagem OK: {img.shape}")
            self._log("2. Executando OCR...")
            result = self.ocr.ocr(img)
            self._log(f"✅ OCR result: {type(result)} len={len(result) if result else 0}")

            if result is None:
                self._log("❌ RESULT IS NONE")
                return []

            first_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
            self._log(f"first_obj type: {type(first_obj)}")

            def get_attr(obj, key):
                if isinstance(obj, dict):
                    return obj.get(key)
                return getattr(obj, key, None)

            rec_texts = get_attr(first_obj, "rec_texts")
            rec_scores = get_attr(first_obj, "rec_scores")
            rec_polys = get_attr(first_obj, "dt_polys") or get_attr(first_obj, "rec_polys")

            self._log(f"rec_texts: {type(rec_texts)} len={len(rec_texts) if rec_texts else 0}")
            self._log(f"rec_scores: {type(rec_scores)} len={len(rec_scores) if rec_scores else 0}")

            lines: List[Dict] = []

            if rec_texts and rec_scores and len(rec_texts) == len(rec_scores):
                self._log("✅ Usando modo rec_texts")
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
                        lines.append({
                            "text": str(text).strip(),
                            "confidence": round(float(confidence), 3),
                            "y_position": y_pos,
                        })
            else:
                self._log("❌ Fallback legado")
                raw_lines = (
                    result[0]
                    if isinstance(result, list)
                    and isinstance(result[0], list)
                    and isinstance(result[0][0], list)
                    else result
                )

                self._log(f"raw_lines type: {type(raw_lines)} len={len(raw_lines) if raw_lines else 0}")

                for item_idx, item in enumerate(raw_lines or []):
                    text = ""
                    confidence = 0.0
                    y_pos = 0

                    if isinstance(item, list) and len(item) >= 2:
                        if isinstance(item[0], list):
                            y_pos = int(item[0][0][1])
                        if isinstance(item[1], (tuple, list)):
                            text = item[1][0]
                            confidence = item[1][1]

                        self._log(f"RAW ITEM {item_idx}: '{text}' conf={confidence}")

                    if text and confidence > 0.4:
                        lines.append({
                            "text": str(text).strip(),
                            "confidence": round(confidence, 3),
                            "y_position": y_pos,
                        })

            lines.sort(key=lambda x: x["y_position"])

            self._log(f"✅ FINAL: {len(lines)} linhas processadas")
            for l_idx, l in enumerate(lines):
                self._log(f"OCR_LINE {l_idx}: y={l['y_position']} | conf={l['confidence']} | '{l['text']}'")

            self._log(f"=== extract_text FINALIZADO ===")
            return lines

        except Exception as e:
            self._log(f"❌ ERRO extract_text: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return []

    # ================ STRUCTURE DATA ================
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

    # ================ PARSER POR LINHA (NFC-e) ================
    def _extract_items_by_line(self, lines: List[Dict], tipo: str) -> List[Dict]:
        data_compra = self._extract_date("\n".join(l.get("text", "") for l in lines))
        itens: List[Dict] = []

        lines_sorted = sorted(
            [l for l in lines if isinstance(l, dict) and "text" in l],
            key=lambda x: x.get("y_position", 0),
        )

        footer_keywords = ["VALOR TOTAL", "QTD. TOTAL", "CARTAO", "CONSUMIDOR", "HTTPS", "PROTOCOLO"]
        valid_lines = [
            l for l in lines_sorted
            if not any(k in l.get("text", "").upper() for k in footer_keywords)
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

            # AUMENTADO: 28 px para agrupar descrição + qtd + preços
            while j < len(valid_lines) and abs(valid_lines[j].get("y_position", 0) - y_base) <= 28:
                block.append(valid_lines[j])
                j += 1

            full_block = " ".join(b.get("text", "") for b in block)
            block_clean = re.sub(r"^\s*\d{1,2}\s+\d{8,14}\s*", "", full_block).strip()

            m = re.search(
                r"""
                (.+?)                          # descrição completa
                \s+(\d+[.,]?\d*)               # quantidade
                \s+(UN|KG|LT|PC|L|M)           # unidade
                (?:\s*[xX]\s*)?                # 'x' opcional
                (\d+[.,]\d{2})                 # valor unitário
                (?:\s+[A-Z0-9]{1,4})?          # T03/F opcional
                \s+(\d+[.,]\d{2})              # valor total
                """,
                block_clean,
                re.IGNORECASE | re.VERBOSE,
            )

            if m:
                desc_raw, qtd, un, v_unit, v_total = m.groups()
                desc = self._clean_desc(desc_raw)
                try:
                    itens.append({
                        "item": desc,
                        "quantidade": float(qtd.replace(",", ".")),
                        "valor_unitario": float(v_unit.replace(",", ".")),
                        "valor_total": float(v_total.replace(",", ".")),
                        "data_compra": data_compra if tipo == "gasto" else None,
                        "data_venda": data_compra if tipo == "venda" else None,
                    })
                except ValueError:
                    pass

            i = j

        return itens


    def _clean_desc(self, desc: str) -> str:
        """Limpa a descrição removendo códigos, unidades, números, caracteres inválidos"""
        if not desc:
            return "ITEM DESCONHECIDO"

        # Remove números no início
        desc = re.sub(r"^\d+\s*", "", desc)
        
        # Remove unidades no final
        desc = re.sub(r"\s+(UN|KG|LT|PC|L|M)\s*$", "", desc, flags=re.IGNORECASE)
        
        # Remove 'x' seguido de valor (ex: x 15,89)
        desc = re.sub(r"\s*[xX]\s*\d+[.,]\d{2}.*$", "", desc)
        
        # Remove códigos de tributo e valores (ex: T03 15,89)
        desc = re.sub(r"\s*(T\d{2,3}|F)\s*\d+[.,]\d{2}.*$", "", desc)
        
        # Normaliza espaços
        desc = re.sub(r"\s+", " ", desc).strip().upper()
        
        # Remove caracteres especiais inválidos
        desc = re.sub(r"[^A-Z0-9À-Ü\s\-\.,/]", "", desc)
        
        # Aplica correções comuns
        for wrong, right in self.COMMON_CORRECTIONS.items():
            if wrong in desc:
                desc = desc.replace(wrong, right)

        return desc if desc else "ITEM DESCONHECIDO"

    # ================ DATA ================
    def _extract_date(self, text: str) -> str:
        """Extrai a data do documento (emissão)"""
        patterns = [
            r"emiss[aã]o[:\s]*(\d{2}/\d{2}/\d{4})",
            r"(\d{2}/\d{2}/\d{4})"
        ]
        
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1)
        
        return datetime.now().strftime("%d/%m/%Y")