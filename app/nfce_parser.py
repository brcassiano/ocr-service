import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class NfceParserSP:
    """
    Parser especializado para NFC-e de São Paulo (portal nfce.fazenda.sp.gov.br).
    """

    # Headers para simular um navegador real e evitar bloqueios simples
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "max-age=0",
    }

    def fetch_html(self, url: str, timeout: int = 25) -> str:
        # Corrige URL se vier com pipes/espaços estranhos do QR Code
        url_clean = url.split('|')[0] if '|' in url else url
        
        logger.info(f"Baixando NFC-e SP: {url_clean[:100]}...")
        
        try:
            resp = requests.get(url_clean, headers=self.HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.error(f"Erro ao baixar URL da SEFAZ: {e}")
            raise

    def parse(self, url: str) -> Dict:
        html = self.fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

        # LOG DE DEBUG: ajuda a ver se a SEFAZ bloqueou ou mandou página de erro
        # Se vier vazio, olhe nos logs o que apareceu aqui.
        title = soup.title.string if soup.title else "Sem Titulo"
        logger.info(f"Página baixada. Título: {title}")
        
        # 1) Verificar se caiu em página de erro/captcha
        if "sessão expirou" in html.lower() or "não encontrado" in html.lower():
            logger.warning("Sessão da SEFAZ expirou ou nota não encontrada.")

        # 2) Extração
        data_compra = self._extract_date(soup)
        itens = self._extract_items_sp(soup, data_compra)
        total_nota = self._extract_total(soup)

        return {
            "tipo_documento": "gasto",
            "itens": itens,
            "total_nota": total_nota,
            "data_compra": data_compra,
            "origem": "nfce_sp_web",
        }

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        # Em SP, geralmente fica num <strong>Emissão:</strong> ou texto solto
        # Tenta procurar padrão de data no topo da página
        for strong in soup.find_all("strong"):
            if "Emissão" in strong.get_text():
                # O pai ou o próximo elemento contém a data
                text = strong.parent.get_text()
                m = re.search(r'(\d{2}/\d{2}/\d{4})', text)
                if m:
                    return m.group(1)
        
        # Fallback: busca qualquer data no corpo
        m = re.search(r'Emissão[:\s]*(\d{2}/\d{2}/\d{4})', soup.get_text(), re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    def _extract_items_sp(self, soup: BeautifulSoup, data_compra: Optional[str]) -> List[Dict]:
        itens = []
        
        # O layout de SP geralmente usa uma tabela com ID "tabResult" 
        # OU uma lista de <tr> com classes específicas como "txtTit", "Rqtd", "valor"
        
        # Estratégia 1: Buscar linhas da tabela de itens (Padrão mais comum)
        rows = soup.find_all("tr", id=re.compile(r"Item"))
        
        if not rows:
            # Estratégia 2: Se não achar pelo ID, busca pela classe da tabela
            table = soup.find("table", {"id": "tabResult"})
            if table:
                rows = table.find_all("tr")

        logger.info(f"Encontradas {len(rows)} linhas potenciais de itens.")

        for row in rows:
            # Nome do Produto (class="txtTit" ou "fixo-prod-serv-descricao")
            nome_elem = row.find(class_="txtTit") or row.find(class_="fixo-prod-serv-descricao")
            if not nome_elem:
                # Tenta pegar primeira célula se for tabela simples
                cols = row.find_all("td")
                if cols: 
                    nome_elem = cols[0]
                else:
                    continue

            nome = nome_elem.get_text(strip=True)
            
            # Pular cabeçalhos
            if "Descrição" in nome or "Qtde" in nome:
                continue

            # Valores e Quantidades (class="Rqtd", "valor")
            # Texto costuma ser: "Qtde.:3" ou "Vl. Unit.:"
            text_row = row.get_text(" ", strip=True)
            
            # Extrair Quantidade
            # Procura "Qtde.: 0,116"
            qtd = 1.0
            m_qtd = re.search(r'Qtde\.?[:\s]*([0-9,.]+)', text_row, re.IGNORECASE)
            if m_qtd:
                qtd = float(m_qtd.group(1).replace(',', '.'))
            
            # Extrair Unitário
            # Procura "Vl. Unit.:" ou "Unit.:"
            valor_unit = None
            m_unit = re.search(r'Unit\.?[:\s]*([0-9,.]+)', text_row, re.IGNORECASE)
            if m_unit:
                valor_unit = float(m_unit.group(1).replace(',', '.'))

            # Extrair Valor Total (geralmente na última coluna ou class="valor")
            valor_total = None
            val_elem = row.find(class_="valor")
            if val_elem:
                valor_total_str = val_elem.get_text(strip=True)
                valor_total = float(valor_total_str.replace(',', '.'))
            else:
                # Tenta pegar último número da linha
                matches = re.findall(r'([0-9]+,[0-9]{2})', text_row)
                if matches:
                    valor_total = float(matches[-1].replace(',', '.'))

            if valor_total is not None:
                itens.append({
                    "item": nome,
                    "quantidade": qtd,
                    "valor_unitario": valor_unit if valor_unit else round(valor_total/qtd, 2),
                    "valor_total": valor_total,
                    "data_compra": data_compra
                })

        logger.info(f"Itens extraídos via Parser SP: {len(itens)}")
        return itens

    def _extract_total(self, soup: BeautifulSoup) -> Optional[float]:
        # Busca classe "txtMax" ou similar onde fica o total
        total_elem = soup.find(class_="txtMax")
        if total_elem:
            try:
                val = total_elem.get_text(strip=True).replace(',', '.')
                return float(val)
            except:
                pass
        
        # Fallback regex
        m = re.search(r'Valor a Pagar.*?([0-9]+,[0-9]{2})', soup.get_text(), re.IGNORECASE)
        if m:
            return float(m.group(1).replace(',', '.'))
        return None