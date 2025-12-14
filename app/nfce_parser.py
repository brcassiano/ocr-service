import re
import logging
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class NfceParserSP:
    """
    Parser NFC-e SP a partir do link do QRCode.

    Estratégia:
    1) Tenta baixar HTML via requests (rápido e barato).
    2) Se não vier tabela/itens ou vier “sessão expirou”, faz fallback via Playwright (render JS).
    """

    def __init__(self, timeout: int = 25):
        self.timeout = timeout
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    # --------- Fetch (STATIC) ----------
    def fetch_html_static(self, url: str) -> str:
        url_clean = url.split("|")[0] if "|" in url else url
        logger.info(f"Baixando HTML (requests) para: {url_clean[:60]}...")

        headers = {"User-Agent": self.user_agent}
        resp = requests.get(url_clean, headers=headers, timeout=self.timeout)
        logger.info(f"HTML len={len(resp.text)} status={resp.status_code}")
        resp.raise_for_status()
        return resp.text

    # --------- Fetch (DYNAMIC) ----------
    async def fetch_html_dynamic(self, url: str) -> str:
        # import atrasado para não quebrar boot se playwright não estiver instalado
        from playwright.async_api import async_playwright

        url_clean = url.split("|")[0] if "|" in url else url
        logger.info(f"Abrindo navegador (Playwright) para: {url_clean[:60]}...")
        logger.info(f"Static signals: table={has_table} item_rows={has_item_rows} tabResult={has_tabresult} txtTit={has_txttit} valor={has_valor}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                context = await browser.new_context(user_agent=self.user_agent)
                page = await context.new_page()

                await page.goto(url_clean, timeout=30000, wait_until="domcontentloaded")
                try:
                    await page.wait_for_selector("table", timeout=10000)
                except Exception:
                    logger.warning("Timeout esperando tabela. Tentando ler o que carregou...")

                html = await page.content()
                return html
            finally:
                await browser.close()

    # --------- Parse ----------
    async def parse(self, url: str) -> Dict:
        # 1) tenta estático
        html = None
        try:
            html = self.fetch_html_static(url)
        except Exception as e:
            logger.warning(f"Falha fetch estático, indo para Playwright: {e}")

        # 2) valida se HTML estático parece “bom”
        if html:
            soup = BeautifulSoup(html, "html.parser")
            if self._is_session_expired(html):
                logger.warning("SEFAZ: Sessão expirou no HTML estático; indo para Playwright.")
                html = None
            else:
                # Se não encontrar nada que pareça itens/tabela, também cai pro Playwright
                has_table = bool(soup.find("table"))
                has_item_rows = bool(soup.find_all("tr", id=re.compile(r"Item")))
                has_tabresult = bool(soup.find("table", {"id": "tabResult"}))
                has_txttit = bool(soup.select_one(".txtTit"))   # título do item
                has_valor = bool(soup.select_one(".valor"))     # valor do item
                if not (has_table or has_item_rows or has_tabresult or has_txttit or has_valor):
                    logger.warning("HTML estático sem sinais de itens; indo para Playwright.")
                    html = None


        # 3) fallback dinâmico
        if not html:
            try:
                html = await self.fetch_html_dynamic(url)
            except ModuleNotFoundError:
                return {
                    "tipo_documento": "erro",
                    "itens": [],
                    "total_nota": None,
                    "data_compra": None,
                    "origem": "nfce_sp_qrcode",
                    "mensagem": "HTML estático não veio parseável e playwright não está instalado.",
                }


        soup = BeautifulSoup(html, "html.parser")

        if self._is_session_expired(html):
            logger.warning("SEFAZ: Sessão expirou mesmo após fallback.")
            return {
                "tipo_documento": "erro",
                "itens": [],
                "total_nota": None,
                "data_compra": None,
                "origem": "nfce_sp",
                "mensagem": "Sessão expirada/consulta bloqueada na SEFAZ-SP",
            }

        data_compra = self._extract_date(soup)
        itens = self._extract_items_sp(soup, data_compra)
        total_nota = self._extract_total(soup)

        return {
            "tipo_documento": "gasto",
            "itens": itens,
            "total_nota": total_nota,
            "data_compra": data_compra,
            "origem": "nfce_sp_qrcode",
        }

    # --------- Helpers ----------
    def _is_session_expired(self, html: str) -> bool:
        h = (html or "").lower()
        return ("sessão expirou" in h) or ("sessao expirou" in h)

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        for strong in soup.find_all("strong"):
            if "Emissão" in strong.get_text():
                text = strong.parent.get_text()
                m = re.search(r"(\d{2}/\d{2}/\d{4})", text)
                if m:
                    return m.group(1)

        m = re.search(r"Emissão[:\s]*(\d{2}/\d{2}/\d{4})", soup.get_text(), re.IGNORECASE)
        if m:
            return m.group(1)

        return None

    def _extract_items_sp(self, soup: BeautifulSoup, data_compra: Optional[str]) -> List[Dict]:
        itens = []

        rows = soup.find_all("tr", id=re.compile(r"Item"))
        if not rows:
            table = soup.find("table", {"id": "tabResult"})
            if table:
                rows = table.find_all("tr")

        for row in rows:
            nome_elem = row.find(class_="txtTit") or row.find(class_="fixo-prod-serv-descricao")
            if not nome_elem:
                continue

            nome = nome_elem.get_text(strip=True)
            if "Descrição" in nome or "Qtde" in nome:
                continue

            text_row = row.get_text(" ", strip=True)

            qtd = 1.0
            m_qtd = re.search(r"Qtde\.?[:\s]*([0-9,.]+)", text_row, re.IGNORECASE)
            if m_qtd:
                try:
                    qtd = float(m_qtd.group(1).replace(",", "."))
                except Exception:
                    qtd = 1.0

            valor_unit = None
            m_unit = re.search(r"Unit\.?[:\s]*([0-9,.]+)", text_row, re.IGNORECASE)
            if m_unit:
                try:
                    valor_unit = float(m_unit.group(1).replace(",", "."))
                except Exception:
                    valor_unit = None

            valor_total = None
            val_elem = row.find(class_="valor")
            if val_elem:
                try:
                    valor_total = float(val_elem.get_text(strip=True).replace(",", "."))
                except Exception:
                    valor_total = None
            else:
                matches = re.findall(r"([0-9]+,[0-9]{2})", text_row)
                if matches:
                    try:
                        valor_total = float(matches[-1].replace(",", "."))
                    except Exception:
                        valor_total = None

            if valor_total is not None:
                itens.append(
                    {
                        "item": nome,
                        "quantidade": qtd,
                        "valor_unitario": valor_unit if valor_unit else round(valor_total / qtd, 2),
                        "valor_total": valor_total,
                        "data_compra": data_compra,
                    }
                )

        return itens

    def _extract_total(self, soup: BeautifulSoup) -> Optional[float]:
        total_elem = soup.find(class_="txtMax")
        if total_elem:
            try:
                return float(total_elem.get_text(strip=True).replace(",", "."))
            except Exception:
                pass

        m = re.search(r"Valor a Pagar.*?([0-9]+,[0-9]{2})", soup.get_text(), re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except Exception:
                return None

        return None