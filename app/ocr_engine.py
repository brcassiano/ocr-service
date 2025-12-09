from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Optional
import logging
from .utils import TextProcessor

print("=" * 80)
print(">>> OCR_ENGINE.PY CARREGADO (VERSÃO NFC-e FIX + DEBUG) <<<")
print("=" * 80)

logger = logging.getLogger(__name__)


class OCREngine:
    """Motor OCR focado em cupons fiscais/NFC-e brasileiros."""

    KEYWORDS_VENDA = ['recebido', 'pix recebido', 'crédito em conta', 'depósito', 'recibo']

    def __init__(self, use_gpu: bool = False):
        logger.info("=" * 60)
        logger.info("INICIALIZANDO OCR ENGINE - VERSÃO NFC-e FIX + DEBUG")
        logger.info("=" * 60)

        try:
            self.ocr = PaddleOCR(
                use_angle_cls=False,   # evita baixar modelo cls extra
                lang='pt',
                use_gpu=use_gpu,
                show_log=False
            )
            logger.info("✓ PaddleOCR inicializado com sucesso")
        except Exception as e:
            logger.error(f"✗ Erro ao inicializar OCR: {e}")
            raise

        self.text_processor = TextProcessor()

    # -------------------------------------------------------------------------
    # QR CODE
    # -------------------------------------------------------------------------
    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        """Extrai QR Code com múltiplas tentativas (pyzbar + OpenCV fallback)."""
        try:
            logger.info("→ Tentando extrair QR Code...")
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("✗ Falha ao decodificar imagem")
                return None

            logger.info(f"  Imagem: {img.shape[1]}x{img.shape[0]} pixels")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1) Tentativas com pyzbar
            attempts = [
                ("grayscale normal", gray),
                ("threshold binário", cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]),
                ("CLAHE contraste", cv2.createCLAHE(
                    clipLimit=3.0, tileGridSize=(8, 8)
                ).apply(gray)),
                ("resize 2x", cv2.resize(
                    gray,
                    (gray.shape[1] * 2, gray.shape[0] * 2),
                    interpolation=cv2.INTER_CUBIC
                )),
            ]

            for method, processed_img in attempts:
                logger.info(f"  Tentativa (pyzbar): {method}")
                decoded = decode(processed_img)
                if decoded:
                    results = []
                    for obj in decoded:
                        if obj.type == 'QRCODE':
                            data = obj.data.decode('utf-8', errors='ignore')
                            results.append({'data': data, 'type': obj.type})
                            logger.info(f"  ✓ QR Code encontrado (pyzbar): {data[:80]}...")
                    if results:
                        return results

            # 2) Fallback com OpenCV QRCodeDetector
            logger.info("  Pyzbar não encontrou nada, tentando OpenCV QRCodeDetector...")
            detector = cv2.QRCodeDetector()
            data, points, _ = detector.detectAndDecode(img)
            if data:
                logger.info(f"  ✓ QR Code encontrado (OpenCV): {data[:80]}...")
                return [{'data': data, 'type': 'QRCODE'}]

            logger.warning("✗ QR Code não detectado (pyzbar nem OpenCV)")
            return None

        except Exception as e:
            logger.error(f"✗ Erro ao extrair QR Code: {e}")
            return None

    # -------------------------------------------------------------------------
    # OCR TEXTO
    # -------------------------------------------------------------------------
    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        """Executa OCR e retorna lista de linhas (texto, confiança, posição)."""
        try:
            logger.info("→ Executando OCR...")
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("✗ Falha ao decodificar imagem")
                return []

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)

            result = self.ocr.ocr(denoised, cls=True)

            if not result or not result[0]:
                logger.warning("✗ OCR não retornou resultados")
                return []

            lines = []
            for line in result[0]:
                text = line[1][0].strip()
                confidence = line[1][1]

                if confidence > 0.4:
                    lines.append({
                        'text': text,
                        'confidence': round(confidence, 3),
                        'y_position': int(line[0][0][1])
                    })
                    logger.info(f"  [{confidence:.0%}] {text}")

            lines.sort(key=lambda x: x['y_position'])
            logger.info(f"✓ OCR concluído: {len(lines)} linhas extraídas")

            # DEBUG: imprimir todas as linhas que o PaddleOCR leu
            print("=== DEBUG OCR LINES ===")
            for i, l in enumerate(lines, 1):
                print(f"{i:02d}: [{l['confidence']}] {l['text']}")
            print("=== FIM DEBUG OCR LINES ===")

            return lines

        except Exception as e:
            logger.error(f"✗ Erro no OCR: {e}")
            return []

    # -------------------------------------------------------------------------
    # ESTRUTURAÇÃO
    # -------------------------------------------------------------------------
    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        """Monta o JSON final a partir das linhas OCR + QRCode."""
        logger.info("→ Estruturando dados...")

        if not ocr_lines:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": None,
                "mensagem": "Não consegui extrair texto.",
                "confianca": 0.0
            }

        full_text = '\n'.join([line['text'] for line in ocr_lines])

        # DEBUG: texto completo plano
        print("=== DEBUG TEXTO COMPLETO ===")
        print(full_text)
        print("=== FIM DEBUG TEXTO COMPLETO ===")

        logger.info(f"=== TEXTO COMPLETO ({len(ocr_lines)} linhas) ===")
        for idx, line in enumerate(ocr_lines[:15], 1):
            logger.info(f"  {idx}. {line['text']}")
        logger.info("=== FIM TEXTO ===")

        tipo = self._detect_type(full_text)
        logger.info(f"  Tipo detectado: {tipo}")

        itens = self._extract_items_nfe_first(ocr_lines, full_text, tipo)
        qr_url = qr_data[0]['data'] if qr_data else None

        avg_conf = sum(l['confidence'] for l in ocr_lines) / len(ocr_lines)

        if not itens:
            logger.error("✗ Nenhum item extraído!")
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": qr_url,
                "mensagem": "Consegui ler mas não encontrei itens válidos.",
                "confianca": round(avg_conf, 3)
            }

        logger.info(f"✓ {len(itens)} itens extraídos com sucesso")
        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_url,
            "mensagem": None,
            "confianca": round(avg_conf, 3)
        }

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def _detect_type(self, text: str) -> str:
        """Detecta se é venda (recebido) ou gasto (compra)."""
        text_lower = text.lower()
        for kw in self.KEYWORDS_VENDA:
            if kw in text_lower:
                logger.info(f"  Keyword venda: '{kw}'")
                return 'venda'
        return 'gasto'

    def _extract_items_nfe_first(self, lines: List[Dict], full_text: str, tipo: str) -> List[Dict]:
        """
        1º tenta extrair itens no padrão NFC-e (linhas 01/02/C2/C3/...).
        Agrupa por bloco de linhas de cada item.
        Se falhar e não for NFC-e, faz um fallback bem conservador.
        """
        logger.info("→ Extraindo itens (priorizando NFC-e)...")

        data_compra = self._extract_date(full_text)
        logger.info(f"  Data detectada: {data_compra}")

        is_nfce = "DOCUMENTO AUXILIAR DA NOTA FISCAL DE CONSUMIDOR" in full_text.upper()
        itens: List[Dict] = []

        # 1) Encontrar cabeçalhos de itens (01, 02, C2, C3, 04...)
        header_indices: List[int] = []
        header_pattern = re.compile(r'^(?:\d{1,2}|C\d)\s+\d{5,}\b')

        for idx, line in enumerate(lines):
            text = " ".join(line['text'].split())
            if header_pattern.match(text):
                header_indices.append(idx)

        if not header_indices:
            logger.warning("  Nenhum cabeçalho de item NFC-e detectado.")
        else:
            logger.info(f"  Cabeçalhos de itens detectados em índices: {header_indices}")

        # 2) Para cada cabeçalho, agrupar linhas e extrair dados
        for i, start in enumerate(header_indices):
            # fim do grupo: antes do próximo cabeçalho ou no máximo 4 linhas adiante
            if i + 1 < len(header_indices):
                end = header_indices[i + 1] - 1
            else:
                end = start + 4
            end = min(end, len(lines) - 1)

            group_lines = lines[start:end + 1]
            header_text = " ".join(group_lines[0]['text'].split())
            other_text = " ".join(l['text'] for l in group_lines[1:])
            logger.info(f"  Grupo de linhas {start}-{end}: '{header_text}' / '{other_text}'")

            # Nome do produto a partir do cabeçalho
            header_clean = re.sub(r'^(?:\d{1,2}|C\d)\s+\d{5,}\s*', '', header_text)
            header_clean = re.sub(r'\s+(KG|UN|LT|L)\s*$', '', header_clean, flags=re.IGNORECASE)
            nome = header_clean.strip(" -")
            if len(nome) < 3:
                nome = "Produto"

            # Quantidade (usar ÚLTIMA ocorrência para evitar MEG8)
            qtd = 1.0
            qtd_matches = list(re.finditer(
                r'([0-9]+[.,]?[0-9]*)\s*(KG|UN|LT|L)\b',
                other_text,
                re.IGNORECASE
            ))
            if qtd_matches:
                m_qtd = qtd_matches[-1]
                qtd = float(m_qtd.group(1).replace(',', '.'))

            # Valor unitário (após 'x ')
            unit_match = re.search(r'x\s*([0-9]+[.,][0-9]{2})', other_text)
            valor_unit = None
            if unit_match:
                valor_unit = float(unit_match.group(1).replace(',', '.'))

            # Candidatos a valores monetários no bloco
            nums = [
                float(v.replace(',', '.'))
                for v in re.findall(r'([0-9]+[.,][0-9]{2})', other_text)
            ]

            valor_total = None
            if valor_unit is not None and nums:
                esperado = qtd * valor_unit
                valor_total = min(nums, key=lambda v: abs(v - esperado))
                logger.info(
                    f"    Valores no grupo: {nums}, esperado={esperado:.2f}, "
                    f"total escolhido={valor_total:.2f}"
                )
            elif nums:
                valor_total = nums[-1]
                logger.info(
                    f"    Sem qty/unit claros, usando último valor do grupo: {valor_total:.2f}"
                )

            if valor_total is None and valor_unit is not None:
                valor_total = round(qtd * valor_unit, 2)

            if valor_total is None:
                logger.warning(
                    f"    Não foi possível determinar total para o item '{nome}', pulando."
                )
                continue

            logger.info(
                f"    ✓ Item NFC-e: nome='{nome}', qtd={qtd}, "
                f"unit={valor_unit}, total={valor_total}"
            )

            itens.append({
                "item": nome,
                "quantidade": qtd,
                "valor_unitario": round(valor_unit, 2) if valor_unit is not None else None,
                "valor_total": round(valor_total, 2),
                "data_compra": data_compra if tipo == 'gasto' else None,
                "data_venda": data_compra if tipo == 'venda' else None,
            })

        # Se é claramente NFC-e, não aplica fallback agressivo
        if is_nfce:
            if not itens:
                logger.warning("  NFC-e detectada mas nenhum item foi reconhecido.")
            return itens

        # ------------------------------------------------------------------
        # Fallback MUITO conservador para outros tipos de comprovante
        # ------------------------------------------------------------------
        all_valores = []
        for line in lines:
            t = line['text']
            if re.search(r'trib\.?|fed:?|est:?|mun:?', t, re.IGNORECASE):
                continue
            valores_linha = re.findall(r'(\d+[.,]\d{2})', t)
            for val in valores_linha:
                try:
                    v = float(val.replace(',', '.'))
                    if 0.01 <= v <= 999999:
                        all_valores.append((v, t))
                except ValueError:
                    pass

        if not all_valores:
            return itens  # vazio

        total_valor = None
        for v, t in all_valores:
            if re.search(r'valor\s+total|total\s*\(r?\$\)|pix', t, re.IGNORECASE):
                total_valor = v
                logger.info(f"  Fallback: total encontrado em linha '{t}' -> {v}")
                break

        if total_valor is None:
            logger.warning("  Fallback: não foi possível determinar TOTAL de forma segura.")
            return itens

        itens.append({
            "item": "Compra",
            "quantidade": 1,
            "valor_unitario": total_valor,
            "valor_total": total_valor,
            "data_compra": data_compra if tipo == 'gasto' else None,
            "data_venda": data_compra if tipo == 'venda' else None,
        })
        return itens

    def _extract_date(self, text: str) -> str:
        """Extrai data preferindo Emissão/Emissao; fallback para qualquer DD/MM/AAAA."""
        patterns = [
            r'emiss[aã]o[:\s]*(\d{2}/\d{2}/\d{4})',
            r'data[:\s]*(\d{2}/\d{2}/\d{4})',
            r'(\d{2}/\d{2}/\d{4})',
        ]

        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                date_str = m.group(1)
                logger.info(f"  Data encontrada: {date_str}")
                return date_str

        hoje = datetime.now().strftime('%d/%m/%Y')
        logger.warning(f"  Data não encontrada, usando hoje: {hoje}")
        return hoje