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
    - tenta requests (às vezes vem completo)
    - fallback Playwright (renderiza JS)
    """

    def __init__(self, timeout: int = 25, enable_debug: bool = False):
        self.timeout = timeout
        self.enable_debug = enable_debug
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    def fetch_html_static(self, url: str) -> str:
        url_clean = url.split("|")[0] if "|" in url else url
        headers = {"User-Agent": self.user_agent}
        resp = requests.get(url_clean, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    async def fetch_html_dynamic(self, url: str) -> str:
        from playwright.async_api import async_playwright

        url_clean = url.split("|")[0] if "|" in url else url
        logger.info(f"[PW] goto: {url_clean[:120]}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                context = await browser.new_context(
                    user_agent=self.user_agent,
                    viewport={"width": 1280, "height": 720},
                    locale="pt-BR",
                )
                page = await context.new_page()

                # "networkidle" costuma ser mais estável que "domcontentloaded"
                await page.goto(url_clean, timeout=60000, wait_until="networkidle")  # [web:180]

                # Espera o TEXTO do DANFE aparecer (mais robusto que selector)
                await page.wait_for_function(
                    """() => {
                        const t = (document.body && document.body.innerText) ? document.body.innerText : "";
                        return t.includes("(Código:") ||
                               t.includes("Valor a pagar") ||
                               t.includes("DOCUMENTO AUXILIAR");
                    }""",
                    timeout=30000,
                )  # [web:177]

                html = await page.content()
                final_url = page.url
                logger.info(f"[PW] final_url={final_url}")
                logger.info(f"[PW] html_len={len(html)}")

                if self.enable_debug:
                    try:
                        txt = await page.locator("body").inner_text()
                        with open("/tmp/nfce_sp.html", "w", encoding="utf-8") as f:
                            f.write(html)
                        with open("/tmp/nfce_sp.txt", "w", encoding="utf-8") as f:
                            f.write(txt)
                        logger.info(f"[PW] body_text_len={len(txt)} head={txt[:250]}")
                    except Exception as e:
                        logger.warning(f"[PW] dump falhou: {e}")

                return html
            finally:
                await browser.close()

    async def parse(self, url: str) -> Dict:
        html: Optional[str] = None
        origem = "nfce_sp_qrcode_static"

        # 1) requests
        try:
            html = self.fetch_html_static(url)
        except Exception as e:
            logger.warning(f"Falha fetch estático: {e}")
            html = None

        if html and not self._is_session_expired(html):
            soup = BeautifulSoup(html, "html.parser")
            data_compra = self._extract_date(soup)
            itens = self._extract_items_sp(soup, data_compra)
            total_nota = self._extract_total(soup)

            if itens:
                out = {
                    "tipo_documento": "gasto",
                    "itens": itens,
                    "total_nota": total_nota,
                    "data_compra": data_compra,
                    "origem": origem,
                }
                if self.enable_debug:
                    out["debug"] = self._debug_block(html, soup, itens)
                return out

        # 2) Playwright
        origem = "nfce_sp_qrcode_browser"
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

        data_compra = self._extract_date(soup)
        itens = self._extract_items_sp(soup, data_compra)
        total_nota = self._extract_total(soup)

        out = {
            "tipo_documento": "gasto",
            "itens": itens,
            "total_nota": total_nota,
            "data_compra": data_compra,
            "origem": origem,
        }
        if self.enable_debug:
            out["debug"] = self._debug_block(html, soup, itens)
        return out

    # ---------------- helpers ----------------
    def _debug_block(self, html: str, soup: BeautifulSoup, itens: List[Dict]) -> Dict:
        page_text = soup.get_text(" ", strip=True)
        return {
            "html_len": len(html or ""),
            "text_len": len(page_text),
            "text_head": page_text[:250],
            "has_codigo": "(Código:" in page_text,
            "has_qtde": "Qtde" in page_text,
            "has_vl_total": "Vl. Total" in page_text,
            "items_found": len(itens),
        }

    def _is_session_expired(self, html: str) -> bool:
        h = (html or "").lower()
        return ("sessão expirou" in h) or ("sessao expirou" in h)

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        txt = soup.get_text(" ", strip=True)
        m = re.search(r"Emissão:\s*(\d{2}/\d{2}/\d{4})", txt, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"\b(\d{2}/\d{2}/\d{4})\s+\d{2}:\d{2}:\d{2}\b", txt)
        if m:
            return m.group(1)
        return None

    def _extract_items_sp(self, soup: BeautifulSoup, data_compra: Optional[str]) -> List[Dict]:
        text = soup.get_text(" ", strip=True)
        text = text.replace("**", " ").replace("|", " ")
        text = re.sub(r"\s+", " ", text).strip()

        itens: List[Dict] = []

        item_re = re.compile(
            r"(?P<desc>.+?)\s*\(Código:\s*(?P<codigo>[^)]+)\)\s*"
            r".*?Qtde\.?:\s*(?P<qtd>[0-9,.]+)\s*"
            r".*?UN:?\s*(?P<un>[A-Z]{1,3})\s*"
            r".*?Vl\.\s*Unit\.?:\s*(?P<vu>[0-9,.]+)\s*"
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
        txt = soup.get_text(" ", strip=True)
        m = re.search(r"Valor a pagar\s*R\$\:?\s*([0-9.,]+)", txt, re.IGNORECASE)
        if m:
            return self._to_float(m.group(1))

        total_elem = soup.find(class_="txtMax")
        if total_elem:
            return self._to_float(total_elem.get_text(strip=True))

        return None

    def _to_float(self, s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s = str(s).strip().replace(" ", "")
        if s.count(",") == 1 and s.count(".") >= 1:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None