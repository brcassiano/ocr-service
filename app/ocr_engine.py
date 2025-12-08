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
    """Motor de OCR para comprovantes fiscais brasileiros"""
    
    # Palavras-chave para detecção de tipo
    KEYWORDS_GASTO = [
        'cupom', 'nota fiscal', 'nfc-e', 'nf-e', 'compra', 'pagamento',
        'débito', 'debito', 'crédito', 'credito', 'cartão', 'cartao',
        'total a pagar', 'valor total', 'subtotal', 'pago',
        'comprovante de pagamento', 'transação aprovada', 'transacao aprovada'
    ]
    
    KEYWORDS_VENDA = [
        'recebido', 'pix recebido', 'crédito em conta', 'credito em conta',
        'depósito', 'deposito', 'transferência recebida', 'transferencia recebida',
        'recibo', 'valor recebido', 'recebi de', 'pelo pagamento de',
        'comprovante de recebimento'
    ]
    
    # Palavras a IGNORAR (não são itens de compra)
    IGNORE_KEYWORDS = [
        'trib', 'tribut', 'aprox', 'fed', 'est', 'mun',
        'total', 'subtotal', 'desconto', 'acrescimo',
        'troco', 'dinheiro', 'cartao', 'cartão',
        'qtd.', 'quantidade', 'codigo', 'sq.codigo',
        'cnpj', 'cpf', 'ie:', 'protocolo', 'autorizacao',
        'emissao', 'consumidor', 'referente', 'pix:',
        'nsu:', 'data:', 'valor:', 'fonte:', 'ibpt'
    ]
    
    def __init__(self, use_gpu: bool = False):
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='pt',
                use_gpu=use_gpu,
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_batch_num=6
            )
            logger.info(f"PaddleOCR inicializado (GPU: {use_gpu})")
        except Exception as e:
            logger.error(f"Erro ao inicializar PaddleOCR: {e}")
            raise
        
        self.text_processor = TextProcessor()
    
    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        """Extrai QR Codes com múltiplas tentativas"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Falha ao decodificar imagem")
                return None
            
            # Tentativa 1: Escala de cinza simples
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            decoded = decode(gray)
            
            # Tentativa 2: Threshold binário
            if not decoded:
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                decoded = decode(binary)
            
            # Tentativa 3: Threshold adaptativo
            if not decoded:
                adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                decoded = decode(adaptive)
            
            # Tentativa 4: Aumentar contraste
            if not decoded:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                decoded = decode(enhanced)
            
            if not decoded:
                logger.info("Nenhum QR Code detectado após 4 tentativas")
                return None
            
            qr_results = []
            for obj in decoded:
                if obj.type == 'QRCODE':
                    try:
                        data = obj.data.decode('utf-8')
                        qr_results.append({
                            'data': data,
                            'type': obj.type,
                            'rect': {
                                'left': obj.rect.left,
                                'top': obj.rect.top,
                                'width': obj.rect.width,
                                'height': obj.rect.height
                            }
                        })
                        logger.info(f"QR Code encontrado: {data[:80]}...")
                    except Exception as e:
                        logger.warning(f"Erro ao decodificar QR: {e}")
            
            return qr_results if qr_results else None
            
        except Exception as e:
            logger.error(f"Erro ao extrair QR Code: {e}")
            return None
    
    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        """Executa OCR completo"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Falha ao decodificar imagem")
                return []
            
            # Pré-processamento
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # OCR
            result = self.ocr.ocr(denoised, cls=True)
            
            if not result or not result[0]:
                logger.warning("OCR não retornou resultados")
                return []
            
            lines = []
            confidences = []
            
            for line in result[0]:
                box = line[0]
                text_data = line[1]
                text = text_data[0]
                confidence = text_data[1]
                
                if confidence > 0.5:
                    y_positions = [point[1] for point in box]
                    avg_y = sum(y_positions) / len(y_positions)
                    
                    lines.append({
                        'text': text.strip(),
                        'confidence': round(confidence, 3),
                        'box': box,
                        'y_position': int(avg_y)
                    })
                    confidences.append(confidence)
            
            lines.sort(key=lambda x: x['y_position'])
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            logger.info(f"OCR: {len(lines)} linhas (confiança: {avg_confidence:.2%})")
            
            return lines
            
        except Exception as e:
            logger.error(f"Erro no OCR: {e}")
            return []
    
    def _detect_document_type(self, text: str) -> str:
        """Detecta tipo de documento"""
        text_lower = text.lower()
        
        venda_score = 0
        gasto_score = 0
        
        for keyword in self.KEYWORDS_VENDA:
            if keyword in text_lower:
                venda_score += 2
        
        for keyword in self.KEYWORDS_GASTO:
            if keyword in text_lower:
                gasto_score += 1
        
        if re.search(r'receb\w*', text_lower):
            venda_score += 3
        
        if re.search(r'pag\w*', text_lower):
            gasto_score += 2
        
        tipo = 'venda' if venda_score > gasto_score else 'gasto'
        logger.info(f"Tipo: {tipo} (venda={venda_score}, gasto={gasto_score})")
        return tipo
    
    def _is_item_line(self, text: str) -> bool:
        """Verifica se linha contém um item de compra válido"""
        text_lower = text.lower()
        
        # Filtrar linhas que devem ser ignoradas
        for keyword in self.IGNORE_KEYWORDS:
            if keyword in text_lower:
                return False
        
        # Linha de item geralmente tem:
        # - Código de produto OU
        # - Nome de produto + valor
        
        # Verificar se tem padrão de item com valor
        # Exemplo: "01 07500435157476 AP GILLETTE FEM C2U 1 UN x 8,48 F 8,48"
        has_price = bool(re.search(r'\d+[.,]\d{2}', text))
        has_quantity = bool(re.search(r'\d+\s*(un|kg|lt|l|pc|pç)', text_lower))
        has_code = bool(re.search(r'^\d{2}\s+\d{8,}', text))  # Código de produto no início
        
        # Se tem código de produto longo + preço, provavelmente é item
        if has_code and has_price:
            return True
        
        # Se tem quantidade e preço, provavelmente é item
        if has_quantity and has_price:
            return True
        
        # Se linha é muito curta (< 10 chars), provavelmente não é item
        if len(text.strip()) < 10:
            return False
        
        return False
    
    def _extract_items(self, lines: List[Dict], tipo: str) -> List[Dict]:
        """Extrai itens com lógica melhorada"""
        if not lines:
            return []
        
        full_text = '\n'.join([line['text'] for line in lines])
        
        # Extrair datas do texto completo
        datas = self.text_processor.extract_dates(full_text)
        data_documento = datas[0] if datas else datetime.now().strftime('%d/%m/%Y')
        
        itens = []
        
        if tipo == 'venda':
            # Para vendas: pegar maior valor
            valores = self.text_processor.extract_money_values(full_text)
            if valores:
                valor_total = max(valores)
                itens.append({
                    "item": None,
                    "quantidade": None,
                    "valor_unitario": None,
                    "valor_total": valor_total,
                    "data_venda": data_documento
                })
                logger.info(f"Venda: R$ {valor_total:.2f}")
        
        else:  # gasto
            # Processar cupom fiscal linha por linha
            total_cupom = None
            
            for line in lines:
                text = line['text']
                
                # Identificar linha de TOTAL
                if re.search(r'(total|pix)\s*\(r\$\)?\s*(\d+[.,]\d{2})', text, re.IGNORECASE):
                    match = re.search(r'(\d+[.,]\d{2})', text)
                    if match:
                        total_str = match.group(1).replace(',', '.')
                        total_cupom = float(total_str)
                        logger.info(f"Total do cupom: R$ {total_cupom:.2f}")
                        continue
                
                # Verificar se é linha de item
                if self._is_item_line(text):
                    item_data = self._parse_item_line(text)
                    if item_data:
                        item_data['data_compra'] = data_documento
                        itens.append(item_data)
                        logger.info(f"Item extraído: {item_data['item']} - R$ {item_data['valor_total']:.2f}")
            
            # Se não encontrou itens, mas encontrou valores
            if not itens:
                valores = self.text_processor.extract_money_values(full_text)
                if valores:
                    # Usar total se encontrou, senão maior valor
                    valor_total = total_cupom if total_cupom else max(valores)
                    
                    itens.append({
                        "item": "Item não identificado",
                        "quantidade": 1,
                        "valor_unitario": valor_total,
                        "valor_total": valor_total,
                        "data_compra": data_documento
                    })
                    logger.warning("Não conseguiu identificar itens individuais")
        
        return itens
    
    def _parse_item_line(self, text: str) -> Optional[Dict]:
        """
        Parse de linha de item do cupom fiscal
        Formato comum: "01 07500435157476 AP GILLETTE FEM C2U 1 UN x 8,48 F 8,48"
        """
        try:
            # Extrair todos os valores monetários da linha
            valores = re.findall(r'(\d+[.,]\d{2})', text)
            if not valores:
                return None
            
            # Último valor geralmente é o total do item
            valor_total_str = valores[-1].replace(',', '.')
            valor_total = float(valor_total_str)
            
            # Extrair quantidade (padrão: número antes de UN/KG/etc)
            quantidade = 1
            qty_match = re.search(r'(\d+)\s*(un|kg|lt|l|pc|pç)', text, re.IGNORECASE)
            if qty_match:
                quantidade = float(qty_match.group(1))
            
            # Calcular valor unitário
            if len(valores) >= 2:
                valor_unit_str = valores[-2].replace(',', '.')
                valor_unitario = float(valor_unit_str)
            else:
                valor_unitario = valor_total / quantidade
            
            # Extrair nome do item (texto entre código e quantidade/valor)
            # Remover código inicial (padrão: 2 dígitos + 13 dígitos)
            text_clean = re.sub(r'^\d{2}\s+\d{8,}', '', text).strip()
            
            # Remover valores e quantidades
            for val in valores:
                text_clean = text_clean.replace(val, '')
            text_clean = re.sub(r'\d+\s*(un|kg|lt|l|pc|pç)', '', text_clean, flags=re.IGNORECASE)
            text_clean = re.sub(r'\s+x\s+', ' ', text_clean)
            text_clean = re.sub(r'\s+[a-z]\s+', ' ', text_clean)  # Remover letras soltas (F, T, etc)
            
            item_name = self.text_processor.clean_item_name(text_clean)
            
            if not item_name or len(item_name) < 3:
                item_name = "Item"
            
            return {
                "item": item_name,
                "quantidade": quantidade,
                "valor_unitario": round(valor_unitario, 2),
                "valor_total": round(valor_total, 2)
            }
            
        except Exception as e:
            logger.error(f"Erro ao fazer parse de item: {e}")
            return None
    
    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        """Estrutura dados finais"""
        if not ocr_lines:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": None,
                "mensagem": "Não consegui extrair texto da imagem. Tente uma foto mais nítida.",
                "confianca": 0.0
            }
        
        full_text = '\n'.join([line['text'] for line in ocr_lines])
        tipo = self._detect_document_type(full_text)
        itens = self._extract_items(ocr_lines, tipo)
        qr_url = qr_data[0]['data'] if qr_data else None
        
        confidences = [line['confidence'] for line in ocr_lines]
        avg_confidence = sum(confidences) / len(confidences)
        
        if not itens:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": qr_url,
                "mensagem": "Consegui ler a imagem, mas não encontrei itens válidos. Tente uma foto mais clara.",
                "confianca": avg_confidence
            }
        
        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_url,
            "mensagem": None,
            "confianca": round(avg_confidence, 3)
        }