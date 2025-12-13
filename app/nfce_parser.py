import re
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class NfceParserSP:
    async def fetch_html_dynamic(self, url: str) -> str:
        # import atrasado: não quebra boot se playwright não estiver instalado
        from playwright.async_api import async_playwright

        url_clean = url.split("|")[0] if "|" in url else url
        logger.info(f"Abrindo navegador (Async) para: {url_clean[:60]}...")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                )
                page = await context.new_page()
                await page.goto(url_clean, timeout=30000, wait_until="domcontentloaded")
                try:
                    await page.wait_for_selector("table", timeout=10000)
                except Exception:
                    logger.warning("Timeout esperando tabela. Tentando ler o que carregou...")
                return await page.content()
            finally:
                await browser.close()

    async def parse(self, url: str) -> Dict:
        html = await self.fetch_html_dynamic(url)
        soup = BeautifulSoup(html, "html.parser")

        data_compra = self._extract_date(soup)
        itens = self._extract_items_sp(soup, data_compra)
        total_nota = self._extract_total(soup)

        return {
            "tipo_documento": "gasto",
            "itens": itens,
            "total_nota": total_nota,
            "data_compra": data_compra,
            "origem": "nfce_sp_browser",
        }

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        m = re.search(r"(\d{2}/\d{2}/\d{4})", soup.get_text(), re.IGNORECASE)
        return m.group(1) if m else None

    def _extract_items_sp(self, soup: BeautifulSoup, data_compra: Optional[str]) -> List[Dict]:
        itens: List[Dict] = []
        rows = soup.find_all("tr", id=re.compile(r"Item", re.IGNORECASE))

        if not rows:
            table = soup.find("table", {"id": "tabResult"})
            if table:
                rows = table.find_all("tr")

        for row in rows:
            nome_elem = row.find(class_="txtTit") or row.find(class_="fixo-prod-serv-descricao")
            if not nome_elem:
                continue

            nome = nome_elem.get_text(strip=True)
            if not nome:
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
        m = re.search(r"(\d+[,\.]\d{2})", soup.get_text(), re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None