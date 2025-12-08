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
    
    # Palavras-chave para detecção de tipo de documento
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
    
    def __init__(self, use_gpu: bool = False):
        """
        Inicializa o motor OCR
        
        Args:
            use_gpu: Se True, usa GPU para processamento (requer CUDA)
        """
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,      # Detecta rotação do texto
                lang='pt',                # Português
                use_gpu=use_gpu,
                show_log=False,
                det_db_thresh=0.3,       # Threshold de detecção
                det_db_box_thresh=0.5,   # Threshold de confiança da box
                rec_batch_num=6          # Processar 6 linhas por vez
            )
            logger.info(f"PaddleOCR inicializado (GPU: {use_gpu})")
        except Exception as e:
            logger.error(f"Erro ao inicializar PaddleOCR: {e}")
            raise
        
        self.text_processor = TextProcessor()
    
    def extract_qrcode(self, image_bytes: bytes) -> Optional[List[Dict]]:
        """
        Extrai QR Codes da imagem
        
        Args:
            image_bytes: Bytes da imagem
            
        Returns:
            Lista de dicts com dados dos QR Codes encontrados, ou None
        """
        try:
            # Converter bytes para numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Falha ao decodificar imagem para QR Code")
                return None
            
            # Pré-processamento para melhorar detecção
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Tentar com imagem normal
            decoded_objects = decode(gray)
            
            # Se não encontrou, tentar com threshold adaptativo
            if not decoded_objects:
                thresh = cv2.adaptiveThreshold(
                    gray, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                decoded_objects = decode(thresh)
            
            if not decoded_objects:
                logger.info("Nenhum QR Code encontrado na imagem")
                return None
            
            # Processar resultados
            qr_results = []
            for obj in decoded_objects:
                if obj.type == 'QRCODE':
                    try:
                        data_decoded = obj.data.decode('utf-8')
                        qr_results.append({
                            'data': data_decoded,
                            'type': obj.type,
                            'rect': {
                                'left': obj.rect.left,
                                'top': obj.rect.top,
                                'width': obj.rect.width,
                                'height': obj.rect.height
                            }
                        })
                        logger.info(f"QR Code encontrado: {data_decoded[:50]}...")
                    except Exception as e:
                        logger.warning(f"Erro ao decodificar QR Code: {e}")
            
            return qr_results if qr_results else None
            
        except Exception as e:
            logger.error(f"Erro ao extrair QR Code: {e}")
            return None
    
    def extract_text(self, image_bytes: bytes) -> List[Dict]:
        """
        Executa OCR completo na imagem
        
        Args:
            image_bytes: Bytes da imagem
            
        Returns:
            Lista de dicts com texto extraído e metadados
        """
        try:
            # Converter para numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Falha ao decodificar imagem para OCR")
                return []
            
            # Pré-processamento para melhorar OCR
            # Converter para escala de cinza
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Executar OCR
            result = self.ocr.ocr(denoised, cls=True)
            
            if not result or not result[0]:
                logger.warning("OCR não retornou resultados")
                return []
            
            # Processar resultados
            lines = []
            confidences = []
            
            for line in result[0]:
                box = line[0]           # Coordenadas [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_data = line[1]     # (texto, confiança)
                text = text_data[0]
                confidence = text_data[1]
                
                # Filtrar resultados de baixa confiança
                if confidence > 0.5:
                    # Calcular posição Y média da box
                    y_positions = [point[1] for point in box]
                    avg_y = sum(y_positions) / len(y_positions)
                    
                    lines.append({
                        'text': text.strip(),
                        'confidence': round(confidence, 3),
                        'box': box,
                        'y_position': int(avg_y)
                    })
                    confidences.append(confidence)
            
            # Ordenar por posição vertical (top to bottom)
            lines.sort(key=lambda x: x['y_position'])
            
            # Calcular confiança média
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            logger.info(f"OCR extraiu {len(lines)} linhas (confiança média: {avg_confidence:.2%})")
            
            return lines
            
        except Exception as e:
            logger.error(f"Erro no OCR: {e}")
            return []
    
    def _detect_document_type(self, text: str) -> str:
        """
        Detecta se documento é gasto ou venda baseado em palavras-chave
        
        Args:
            text: Texto completo extraído
            
        Returns:
            'gasto', 'venda' ou 'gasto' (default)
        """
        text_lower = text.lower()
        
        # Contadores de matches
        gasto_score = 0
        venda_score = 0
        
        # Verificar keywords de venda (peso maior)
        for keyword in self.KEYWORDS_VENDA:
            if keyword in text_lower:
                venda_score += 2
                logger.debug(f"Keyword venda encontrada: '{keyword}'")
        
        # Verificar keywords de gasto
        for keyword in self.KEYWORDS_GASTO:
            if keyword in text_lower:
                gasto_score += 1
                logger.debug(f"Keyword gasto encontrada: '{keyword}'")
        
        # Heurísticas adicionais
        
        # Se tem "receb" (recebido, recebimento, etc), provavelmente é venda
        if re.search(r'receb\w*', text_lower):
            venda_score += 3
        
        # Se tem "pag" (pagamento, pago, pagar), provavelmente é gasto
        if re.search(r'pag\w*', text_lower):
            gasto_score += 2
        
        # Detectar comprovantes de transferência/PIX
        if any(word in text_lower for word in ['pix', 'ted', 'doc', 'transferência', 'transferencia']):
            # Se tem "enviado" ou "para", é gasto
            if any(word in text_lower for word in ['enviado', 'para:', 'destinatário', 'destinatario']):
                gasto_score += 2
            # Se tem "recebido" ou "de", é venda
            elif any(word in text_lower for word in ['recebido', 'de:', 'remetente', 'origem']):
                venda_score += 3
        
        # Decisão final
        if venda_score > gasto_score:
            tipo = 'venda'
        else:
            tipo = 'gasto'  # Default para gasto
        
        logger.info(f"Tipo detectado: {tipo} (venda_score={venda_score}, gasto_score={gasto_score})")
        return tipo
    
    def _extract_items(self, lines: List[Dict], tipo: str) -> List[Dict]:
        """
        Extrai itens estruturados das linhas OCR
        
        Args:
            lines: Lista de linhas extraídas pelo OCR
            tipo: Tipo do documento ('gasto' ou 'venda')
            
        Returns:
            Lista de itens estruturados
        """
        if not lines:
            return []
        
        # Concatenar texto completo
        full_text = '\n'.join([line['text'] for line in lines])
        
        # Extrair valores e datas
        valores = self.text_processor.extract_money_values(full_text)
        datas = self.text_processor.extract_dates(full_text)
        
        # Pegar primeira data encontrada ou usar data atual
        data_documento = datas[0] if datas else datetime.now().strftime('%d/%m/%Y')
        
        if not valores:
            logger.warning("Nenhum valor monetário encontrado")
            return []
        
        itens = []
        
        if tipo == 'venda':
            # Para vendas: normalmente 1 único lançamento
            valor_total = max(valores)  # Maior valor é geralmente o total
            
            itens.append({
                "item": None,
                "quantidade": None,
                "valor_unitario": None,
                "valor_total": valor_total,
                "data_venda": data_documento
            })
            
            logger.info(f"Venda extraída: R$ {valor_total:.2f}")
        
        else:  # gasto
            # Para gastos: tentar identificar múltiplos itens
            
            # Se encontrou apenas 1 ou 2 valores, criar item único
            if len(valores) <= 2:
                valor_total = max(valores)
                
                # Tentar identificar nome do item
                item_name = self._find_item_name_near_value(lines, valor_total)
                
                itens.append({
                    "item": item_name or "Item",
                    "quantidade": 1,
                    "valor_unitario": valor_total,
                    "valor_total": valor_total,
                    "data_compra": data_documento
                })
            
            else:
                # Múltiplos valores: último geralmente é o total
                total_provavel = max(valores)
                valores_itens = [v for v in valores if v < total_provavel]
                
                # Se não sobrou nada, usar todos
                if not valores_itens:
                    valores_itens = valores[:-1]  # Todos exceto o último
                
                # Criar itens
                for idx, valor in enumerate(valores_itens, 1):
                    item_name = self._find_item_name_near_value(lines, valor)
                    
                    itens.append({
                        "item": item_name or f"Item {idx}",
                        "quantidade": 1,
                        "valor_unitario": valor,
                        "valor_total": valor,
                        "data_compra": data_documento
                    })
                
                logger.info(f"Gasto extraído: {len(itens)} itens, total R$ {total_provavel:.2f}")
        
        return itens
    
    def _find_item_name_near_value(self, lines: List[Dict], valor: float) -> Optional[str]:
        """
        Tenta encontrar nome do item próximo ao valor monetário
        
        Args:
            lines: Linhas do OCR
            valor: Valor monetário a procurar
            
        Returns:
            Nome do item ou None
        """
        valor_str_patterns = [
            f"{valor:.2f}",                          # 10.50
            f"{valor:,.2f}".replace(",", "."),      # 1.000.50
            f"R$ {valor:.2f}",
            f"R${valor:.2f}",
        ]
        
        # Procurar linha que contém o valor
        for i, line in enumerate(lines):
            line_text = line['text']
            
            # Verificar se linha contém o valor
            if any(pattern in line_text for pattern in valor_str_patterns):
                # Pegar texto antes do valor na mesma linha
                for pattern in valor_str_patterns:
                    if pattern in line_text:
                        item_name = line_text.split(pattern)[0].strip()
                        if item_name and len(item_name) > 2:
                            return self.text_processor.clean_item_name(item_name)
                
                # Se não achou na mesma linha, tentar linha anterior
                if i > 0:
                    prev_line = lines[i-1]['text']
                    if not any(char.isdigit() for char in prev_line):  # Não contém números
                        return self.text_processor.clean_item_name(prev_line)
        
        return None
    
    def structure_data(self, ocr_lines: List[Dict], qr_data: Optional[List[Dict]]) -> Dict:
        """
        Estrutura dados extraídos em formato JSON final
        
        Args:
            ocr_lines: Linhas extraídas pelo OCR
            qr_data: Dados do QR Code (se houver)
            
        Returns:
            Dict com estrutura final
        """
        if not ocr_lines:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": None,
                "mensagem": "Não consegui extrair texto da imagem. Tente enviar uma foto mais nítida.",
                "confianca": 0.0
            }
        
        # Texto completo
        full_text = '\n'.join([line['text'] for line in ocr_lines])
        
        # Detectar tipo
        tipo = self._detect_document_type(full_text)
        
        # Extrair itens
        itens = self._extract_items(ocr_lines, tipo)
        
        # QR Code URL
        qr_url = qr_data[0]['data'] if qr_data else None
        
        # Confiança média
        confidences = [line['confidence'] for line in ocr_lines]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Validar se conseguiu extrair dados mínimos
        if not itens:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "qrcode_url": qr_url,
                "mensagem": "Consegui ler a imagem, mas não encontrei valores monetários. Tente uma foto mais clara dos valores.",
                "confianca": avg_confidence
            }
        
        return {
            "tipo_documento": tipo,
            "itens": itens,
            "qrcode_url": qr_url,
            "mensagem": None,
            "confianca": round(avg_confidence, 3)
        }