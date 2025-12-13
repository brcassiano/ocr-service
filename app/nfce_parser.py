import re
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class NfceParserSP:
    """
    Parser usando Playwright Async para renderizar JS da SEFAZ-SP.
    IMPORTANTE: Playwright é importado apenas quando necessário para não quebrar o boot.
    """

    async def fetch_html_dynamic(self, url: str) -> str:
        # Import atrasado: evita ModuleNotFoundError ao importar o módulo sem playwright instalado
        from playwright.async_api import async_playwright

        url_clean = url.split("|")[0] if "|" in url else url
        logger.info(f"Abrindo navegador Async para {url_clean}...")

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

                # tenta esperar tabela; se não, pega o que tiver
                try:
                    await page.wait_for_selector("table", timeout=10000)
                except Exception:
                    logger.warning("Timeout esperando tabela; retornando HTML disponível.")

                html = await page.content()
                return html
            finally:
                await browser.close()

    async def parse(self, url: str) -> Dict:
        html = await self.fetch_html_dynamic(url)

        soup = BeautifulSoup(html, "html.parser")
        if "sessão expirou" in html.lower() or "sessao expirou" in html.lower():
            logger.warning("SEFAZ: sessão expirou.")

        data_compra = self.extract_date(soup)
        itens = self.extract_items_sp(soup, data_compra)
        total_nota = self.extract_total(soup)

        return {
            "tipo_documento": "gasto",
            "itens": itens,
            "total_nota": total_nota,
            "data_compra": data_compra,
            "origem": "nfce-sp-browser",
        }

    def extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        # tenta achar "Emissão"
        text = soup.get_text(" ", strip=True)
        m = re.search(r"(\d{2}/\d{2}/\d{4})", text)
        return m.group(1) if m else None

    def extract_items_sp(self, soup: BeautifulSoup, data_compra: Optional[str]) -> List[Dict]:
        itens = []

        rows = soup.find_all("tr", id=re.compile(r"Item", re.IGNORECASE))
        if not rows:
            table = soup.find("table", id="tabResult")
            if table:
                rows = table.find_all("tr")

        for row in rows:
            nome_elem = row.find(class_="txtTit") or row.find(class_="fixo-prod-serv-descricao")
            if not nome_elem:
                continue

            nome = nome_elem.get_text(strip=True)
            if not nome or "Descrição" in nome or "Qtde" in nome:
                continue

            text_row = row.get_text(" ", strip=True)

            qtd = 1.0
            mqtd = re.search(r"Qtde\.?\s*([\d,\.]+)", text_row, re.IGNORECASE)
            if mqtd:
                try:
                    qtd = float(mqtd.group(1).replace(",", "."))
                except Exception:
                    qtd = 1.0

            valor_unit = None
            munit = re.search(r"Unit\.?\s*([\d,\.]+)", text_row, re.IGNORECASE)
            if munit:
                try:
                    valor_unit = float(munit.group(1).replace(",", "."))
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
                matches = re.findall(r"(\d+[,\.]\d{2})", text_row)
                if matches:
                    try:
                        valor_total = float(matches[-1].replace(",", "."))
                    except Exception:
                        valor_total = None

            if valor_total is None:
                continue

            itens.append(
                {
                    "item": nome,
                    "quantidade": qtd,
                    "valor_unitario": valor_unit if valor_unit is not None else round(valor_total / qtd, 2),
                    "valor_total": valor_total,
                    "data_compra": data_compra,
                }
            )

        return itens

    def extract_total(self, soup: BeautifulSoup) -> Optional[float]:
        total_elem = soup.find(class_="txtMax")
        if total_elem:
            try:
                return float(total_elem.get_text(strip=True).replace(",", "."))
            except Exception:
                pass

        text = soup.get_text(" ", strip=True)
        m = re.search(r"Valor a Pagar.*?(\d+[,\.]\d{2})", text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except Exception:
                return None
        return None