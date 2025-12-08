import re
from datetime import datetime
from typing import List, Tuple

class TextProcessor:
    """Utilitários para processamento de texto extraído"""
    
    @staticmethod
    def extract_money_values(text: str) -> List[float]:
        """
        Extrai valores monetários do texto.
        Suporta formatos: R$ 10,50 | 10.50 | 10,50 | R$10,50
        """
        # Padrões de valores monetários
        patterns = [
            r'R\$?\s*(\d{1,3}(?:\.\d{3})*,\d{2})',  # R$ 1.000,50
            r'R\$?\s*(\d+,\d{2})',                    # R$ 50,00 ou R$50,00
            r'\b(\d+\.\d{2})\b',                      # 50.00 (formato float)
        ]
        
        valores = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Normalizar para float
                    if ',' in match:
                        # Formato brasileiro: 1.000,50 -> 1000.50
                        valor_clean = match.replace('.', '').replace(',', '.')
                    else:
                        # Já está em formato correto
                        valor_clean = match
                    
                    valor_float = float(valor_clean)
                    if 0.01 <= valor_float <= 999999.99:  # Validar range razoável
                        valores.append(valor_float)
                except ValueError:
                    continue
        
        return sorted(set(valores))  # Remover duplicatas e ordenar
    
    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """
        Extrai datas do texto.
        Suporta: DD/MM/YYYY, DD/MM/YY, DD-MM-YYYY
        """
        patterns = [
            r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b',  # DD/MM/YYYY ou DD-MM-YYYY
            r'\b(\d{2}[/-]\d{2}[/-]\d{2})\b',  # DD/MM/YY
        ]
        
        datas = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                normalized = TextProcessor.normalize_date(match)
                if normalized:
                    datas.append(normalized)
        
        return datas
    
    @staticmethod
    def normalize_date(date_str: str) -> str:
        """Normaliza data para formato DD/MM/YYYY"""
        try:
            # Substituir - por /
            date_str = date_str.replace('-', '/')
            parts = date_str.split('/')
            
            if len(parts) != 3:
                return datetime.now().strftime('%d/%m/%Y')
            
            day, month, year = parts
            
            # Validar dia e mês
            if not (1 <= int(day) <= 31 and 1 <= int(month) <= 12):
                return datetime.now().strftime('%d/%m/%Y')
            
            # Expandir ano de 2 dígitos
            if len(year) == 2:
                year = f"20{year}"
            
            return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
        except:
            return datetime.now().strftime('%d/%m/%Y')
    
    @staticmethod
    def clean_item_name(text: str) -> str:
        """Limpa nome de item removendo valores e caracteres especiais"""
        # Remover valores monetários
        text = re.sub(r'R\$?\s*[\d.,]+', '', text, flags=re.IGNORECASE)
        # Remover números soltos no final
        text = re.sub(r'\s+\d+$', '', text)
        # Remover caracteres especiais excessivos
        text = re.sub(r'[*]{2,}', '', text)
        return text.strip()