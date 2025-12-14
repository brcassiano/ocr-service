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
    2) Faz parse dos itens do HTML.
    3) Só se não encontrar itens (ou sessão expirada) tenta Playwright.
       Se Playwright não estiver instalado, retorna erro amigável.
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
        resp.raise_for_status()

        logger.info(f"HTML estático baixado: status={resp.status_code} len={len(resp.text)}")
        return resp.text

    # --------- Fetch (DYNAMIC) ----------
    async def fetch_html_dynamic(self, url: str) -> str:
        # import atrasado (só se precisar)
        from playwright.async_api import async_playwright

        url_clean = url.split("|")[0] if "|" in url else url
        logger.info(f"Abrindo navegador (Playwright) para: {url_clean[:60]}...")

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
                logger.info(f"HTML dinâmico carregado len={len(html)}")
                return html
            finally:
                await browser.close()

    # --------- Parse ----------
    async def parse(self, url: str) -> Dict:
        html = None

        # 1) tenta requests
        try:
            html = self.fetch_html_static(url)
        except Exception as e:
            logger.warning(f"Falha fetch estático: {e}")
            html = None

        # 2) tenta parsear itens do HTML estático (se veio)
        if html and not self._is_session_expired(html):
            soup = BeautifulSoup(html, "html.parser")

            # sinais “reais” de que a página tem itens
            signals = {
                "txtTit": bool(soup.select_one(".txtTit")),
                "valor": bool(soup.select_one(".valor")),
                "codigo_text": "(Código:" in soup.get_text(" ", strip=True),
                "vl_total_text": "Vl. Total" in soup.get_text(" ", strip=True),
            }
            logger.info(f"Static signals: {signals}")

            data_compra = self._extract_date(soup)
            itens = self._extract_items_sp(soup, data_compra)
            total_nota = self._extract_total(soup)

            # Se achou itens, já retorna e não tenta Playwright
            if itens:
                return {
                    "tipo_documento": "gasto",
                    "itens": itens,
                    "total_nota": total_nota,
                    "data_compra": data_compra,
                    "origem": "nfce_sp_qrcode_static",
                }

        # 3) se sessão expirada ou não achou itens, tenta Playwright (se existir)
        try:
            html = await self.fetch_html_dynamic(url)
        except ModuleNotFoundError:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "total_nota": None,
                "data_compra": None,
                "origem": "nfce_sp_qrcode",
                "mensagem": "HTML estático não retornou itens e playwright não está instalado.",
            }
        except Exception as e:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "total_nota": None,
                "data_compra": None,
                "origem": "nfce_sp_qrcode",
                "mensagem": f"Falha ao consultar via Playwright: {str(e)}",
            }

        soup = BeautifulSoup(html, "html.parser")
        if self._is_session_expired(html):
            return {
                "tipo_documento": "erro",
                "itens": [],
                "total_nota": None,
                "data_compra": None,
                "origem": "nfce_sp_qrcode",
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
            "origem": "nfce_sp_qrcode_browser",
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
        itens: List[Dict] = []

        rows = soup.find_all("tr", id=re.compile(r"Item"))
        if not rows:
            table = soup.find("table", {"id": "tabResult"})
            if table:
                rows = table.find_all("tr")
        if not rows:
            rows = soup.find_all("tr")

        for row in rows:
            text_row = row.get_text(" ", strip=True)
            if not text_row:
                continue

            # precisa ter algum indício de item
            if "(Código:" not in text_row and "Vl. Total" not in text_row and "Vl.Total" not in text_row:
                continue

            nome_elem = row.find(class_="txtTit") or row.find(class_="fixo-prod-serv-descricao")
            nome = nome_elem.get_text(strip=True) if nome_elem else None

            # fallback: quando não existir txtTit, corta antes do "(Código:"
            if not nome and "(Código:" in text_row:
                nome = text_row.split("(Código:")[0].strip()

            if not nome:
                continue
            if "Descrição" in nome or "Qtde" in nome:
                continue

            # quantidade
            qtd = 1.0
            m_qtd = re.search(r"Qtde\.?\:?\s*([0-9,.]+)", text_row, re.IGNORECASE)
            if m_qtd:
                try:
                    qtd = float(m_qtd.group(1).replace(",", "."))
                except Exception:
                    qtd = 1.0

            # valor unitário (padrão SP: "Vl. Unit.")
            valor_unit = None
            m_unit = re.search(r"Vl\.\s*Unit\.\:?\s*([0-9,.]+)", text_row, re.IGNORECASE)
            if not m_unit:
                m_unit = re.search(r"Unit\.?\:?\s*([0-9,.]+)", text_row, re.IGNORECASE)
            if m_unit:
                try:
                    valor_unit = float(m_unit.group(1).replace(",", "."))
                except Exception:
                    valor_unit = None

            # valor total (preferir "Vl. Total")
            valor_total = None
            m_total = re.search(r"Vl\.\s*Total\s*([0-9,.]+)", text_row, re.IGNORECASE)
            if m_total:
                try:
                    valor_total = float(m_total.group(1).replace(",", "."))
                except Exception:
                    valor_total = None
            else:
                matches = re.findall(r"([0-9]+,[0-9]{2})", text_row)
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