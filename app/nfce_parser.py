# app/nfce_parser.py
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class NfceParserSP:
    """
    Parser simples para NFC-e de São Paulo a partir da URL pública
    (link do QR Code, ex: https://www.nfce.fazenda.sp.gov.br/NFCeConsultaPublica/...).
    """

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )

    def fetch_html(self, url: str, timeout: int = 20) -> str:
        headers = {"User-Agent": self.USER_AGENT}
        logger.info(f"Baixando NFC-e: {url[:120]}...")
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text

    def parse(self, url: str) -> Dict:
        """
        Retorna um dicionário com itens, data, total e alguns metadados.
        Estrutura orientada ao seu fluxo atual.
        """
        html = self.fetch_html(url)
        soup = BeautifulSoup(html, "lxml")

        # 1) Data da emissão (formato varia; normalmente tem "Emissão:" ou similar)
        data_compra = self._extract_date(soup)

        # 2) Itens – tabela principal de produtos
        itens = self._extract_items(soup, data_compra)

        # 3) Total da NFC-e
        total_nota = self._extract_total(soup)

        return {
            "tipo_documento": "gasto",
            "itens": itens,
            "total_nota": total_nota,
            "data_compra": data_compra,
            "origem": "nfce_sp",
        }

    # ------------------------------------------------------------------ helpers

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        # Procura qualquer texto que tenha padrão DD/MM/AAAA
        text = soup.get_text(" ", strip=True)
        m = re.search(r'(\d{2}/\d{2}/\d{4})', text)
        if m:
            return m.group(1)
        return None

    def _extract_items(self, soup: BeautifulSoup, data_compra: Optional[str]) -> List[Dict]:
        itens: List[Dict] = []

        # A tabela de itens na NFC-e SP costuma ter cabeçalhos como "CÓDIGO", "DESCRIÇÃO", "QTD", etc. [web:103][web:113]
        # Procura todas as tabelas e filtra pela presença desses cabeçalhos.
        candidate_tables = []
        for table in soup.find_all("table"):
            header_text = " ".join(table.get_text(" ", strip=True).upper().split())
            if ("DESCRI" in header_text and "QTD" in header_text) or ("QUANT" in header_text and "UNIT" in header_text):
                candidate_tables.append(table)

        if not candidate_tables:
            logger.warning("Nenhuma tabela de itens NFC-e encontrada.")
            return itens

        table = candidate_tables[0]  # pega a primeira candidata

        # Normalmente as linhas de itens estão em <tr> depois do cabeçalho.
        for tr in table.find_all("tr"):
            cols = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cols) < 4:
                continue

            linha = " ".join(cols)
            # Heurística: pula linha de cabeçalho.
            if "DESCRI" in cols[0].upper() or "CÓDIGO" in cols[0].upper():
                continue

            # Tenta separar: código, descrição, quantidade, unitário, total
            # Cada UF varia; aqui usamos regex como fallback.
            descricao = " ".join(cols[1:-3]) if len(cols) >= 5 else cols[1]

            # Tenta encontrar números decimais na linha (últimos dois normalmente são unitário e total)
            valores = re.findall(r'(\d+[.,]\d{2})', linha)
            valor_unit = None
            valor_total = None
            if len(valores) >= 2:
                valor_unit = float(valores[-2].replace(",", "."))
                valor_total = float(valores[-1].replace(",", "."))
            elif len(valores) == 1:
                valor_total = float(valores[0].replace(",", "."))

            # Quantidade: procura número antes de "UN", "KG", etc.
            qtd = 1.0
            m_qtd = re.search(r'(\d+[.,]?\d*)\s*(UN|KG|LT|L)\b', linha, re.IGNORECASE)
            if m_qtd:
                qtd = float(m_qtd.group(1).replace(",", "."))

            if valor_total is None:
                continue

            if not descricao or len(descricao) < 3:
                descricao = "Produto"

            itens.append({
                "item": descricao,
                "quantidade": qtd,
                "valor_unitario": round(valor_unit, 2) if valor_unit is not None else None,
                "valor_total": round(valor_total, 2),
                "data_compra": data_compra,
            })

        logger.info(f"Itens extraídos da NFC-e: {len(itens)}")
        return itens

    def _extract_total(self, soup: BeautifulSoup) -> Optional[float]:
        text = soup.get_text(" ", strip=True).upper()
        # Procura algo como "VALOR TOTAL R$ 19,78" [web:103]
        m = re.search(r'VALOR\s+TOTAL[^\d]*([\d.,]{3,})', text)
        if m:
            try:
                return float(m.group(1).replace(".", "").replace(",", "."))
            except ValueError:
                return None
        return None