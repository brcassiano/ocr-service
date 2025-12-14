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
    - requests -> parse HTML
    - extrai itens por DOM; se falhar, extrai por regex no texto completo.
    - Playwright vira apenas fallback opcional (se instalado).
    """

    def __init__(self, timeout: int = 25):
        self.timeout = timeout
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    def fetch_html_static(self, url: str) -> str:
        url_clean = url.split("|")[0] if "|" in url else url
        logger.info(f"Baixando HTML (requests) para: {url_clean[:60]}...")

        headers = {"User-Agent": self.user_agent}
        resp = requests.get(url_clean, headers=headers, timeout=self.timeout)
        resp.raise_for_status()

        logger.info(f"HTML estático baixado: status={resp.status_code} len={len(resp.text)}")
        return resp.text

    async def fetch_html_dynamic(self, url: str) -> str:
        from playwright.async_api import async_playwright  # import atrasado

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

    async def parse(self, url: str) -> Dict:
        # 1) requests
        try:
            html = self.fetch_html_static(url)
        except Exception as e:
            logger.warning(f"Falha fetch estático: {e}")
            html = None

        # 2) tenta parse no estático
        if html and not self._is_session_expired(html):
            soup = BeautifulSoup(html, "html.parser")
            data_compra = self._extract_date(soup)
            itens = self._extract_items_sp(soup, data_compra)
            total_nota = self._extract_total(soup)

            if itens:
                return {
                    "tipo_documento": "gasto",
                    "itens": itens,
                    "total_nota": total_nota,
                    "data_compra": data_compra,
                    "origem": "nfce_sp_qrcode_static",
                }

        # 3) fallback Playwright (se existir)
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

    # ---------------- helpers ----------------
    def _is_session_expired(self, html: str) -> bool:
        h = (html or "").lower()
        return ("sessão expirou" in h) or ("sessao expirou" in h)

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        # Emissão: 11/12/2025 18:57:55
        m = re.search(r"Emissão:\s*(\d{2}/\d{2}/\d{4})", soup.get_text(" ", strip=True), re.IGNORECASE)
        if m:
            return m.group(1)

        # fallback antigo
        for strong in soup.find_all("strong"):
            if "Emissão" in strong.get_text():
                text = strong.parent.get_text()
                m2 = re.search(r"(\d{2}/\d{2}/\d{4})", text)
                if m2:
                    return m2.group(1)

        return None

    def _extract_items_sp(self, soup: BeautifulSoup, data_compra: Optional[str]) -> List[Dict]:
        """
        Tenta 2 estratégias:
        A) DOM (tr/td + classes txtTit/valor)
        B) Regex no texto completo (muito robusto pro layout SP)
        """
        itens: List[Dict] = []

        # ---------- A) DOM ----------
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

            # indícios
            if "(Código:" not in text_row and "Vl. Total" not in text_row and "Vl.Total" not in text_row:
                continue

            nome_elem = row.find(class_="txtTit") or row.find(class_="fixo-prod-serv-descricao")
            nome = nome_elem.get_text(strip=True) if nome_elem else None
            if not nome and "(Código:" in text_row:
                nome = text_row.split("(Código:")[0].strip()

            if not nome or "Descrição" in nome or "Qtde" in nome:
                continue

            qtd = self._extract_float(text_row, r"Qtde\.?\:?\s*([0-9,.]+)") or 1.0
            valor_unit = self._extract_float(text_row, r"Vl\.\s*Unit\.\:?\s*([0-9,.]+)")
            if valor_unit is None:
                valor_unit = self._extract_float(text_row, r"Unit\.?\:?\s*([0-9,.]+)")

            valor_total = self._extract_float(text_row, r"Vl\.\s*Total\s*([0-9,.]+)")
            if valor_total is None:
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

        if itens:
            return itens

        # ---------- B) Regex no texto completo ----------
        text = soup.get_text(" ", strip=True)

        # padrão do seu HTML:
        # "BISNAG PANCO 300G (Código: 7891203010605 ) Qtde.: 1 UN: UN Vl. Unit.: 8,99 | Vl. Total 8,99"
        item_re = re.compile(
            r"(?P<desc>.+?)\s*\(Código:\s*(?P<codigo>[^)]+)\)\s*"
            r".*?Qtde\.\:\s*(?P<qtd>[0-9,.]+)\s*"
            r".*?UN:\s*(?P<un>[A-Z]+)\s*"
            r".*?Vl\.\s*Unit\.\:\s*(?P<vu>[0-9,.]+)\s*"
            r".*?Vl\.\s*Total\s*(?P<vt>[0-9,.]+)",
            re.IGNORECASE,
        )

        for m in item_re.finditer(text):
            desc = m.group("desc").strip()
            qtd = self._to_float(m.group("qtd")) or 1.0
            vu = self._to_float(m.group("vu"))
            vt = self._to_float(m.group("vt"))
            if vt is None:
                continue

            itens.append(
                {
                    "item": desc,
                    "quantidade": qtd,
                    "valor_unitario": vu if vu is not None else round(vt / qtd, 2),
                    "valor_total": vt,
                    "data_compra": data_compra,
                }
            )

        return itens

    def _extract_total(self, soup: BeautifulSoup) -> Optional[float]:
        # "Valor a pagar R$:236,09"
        m = re.search(r"Valor a pagar\s*R\$\:\s*([0-9,.]+)", soup.get_text(" ", strip=True), re.IGNORECASE)
        if m:
            return self._to_float(m.group(1))

        total_elem = soup.find(class_="txtMax")
        if total_elem:
            try:
                return float(total_elem.get_text(strip=True).replace(",", "."))
            except Exception:
                pass

        return None

    def _extract_float(self, text: str, pattern: str) -> Optional[float]:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return None
        return self._to_float(m.group(1))

    def _to_float(self, s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        try:
            return float(str(s).strip().replace(".", "").replace(",", "."))
        except Exception:
            return None