import re
import logging
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class NfceParserSP:
    def __init__(self, timeout: int = 25, enable_debug: bool = False):
        self.timeout = timeout
        self.enable_debug = enable_debug
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    def _clean_url(self, url: str) -> str:
        return url.split("|")[0] if "|" in url else url

    def fetch_html_static(self, url: str) -> str:
        url_clean = self._clean_url(url)
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
        }
        resp = requests.get(url_clean, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    async def fetch_html_dynamic(self, url: str) -> str:
        from playwright.async_api import async_playwright

        url_clean = self._clean_url(url)

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",  # stealth-lite [web:191]
                ],
            )
            try:
                context = await browser.new_context(
                    user_agent=self.user_agent,
                    locale="pt-BR",
                    viewport={"width": 1280, "height": 720},
                )

                # esconde webdriver (stealth-lite) [web:193]
                await context.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                )

                page = await context.new_page()
                await page.goto(url_clean, timeout=60000, wait_until="domcontentloaded")

                # espera mais “humana”: dá tempo do JS preencher a tela
                await page.wait_for_timeout(2500)

                html = await page.content()

                if self.enable_debug:
                    try:
                        txt = await page.locator("body").inner_text()
                        with open("/tmp/nfce_sp_pw.html", "w", encoding="utf-8") as f:
                            f.write(html)
                        with open("/tmp/nfce_sp_pw.txt", "w", encoding="utf-8") as f:
                            f.write(txt)
                        logger.info(f"[PW] final_url={page.url} body_text_len={len(txt)}")
                    except Exception as e:
                        logger.warning(f"[PW] dump falhou: {e}")

                return html
            finally:
                await browser.close()

    async def parse(self, url: str) -> Dict:
        # 1) tenta requests (melhor quando o Playwright é bloqueado) [web:1]
        html_static = None
        try:
            html_static = self.fetch_html_static(url)
            soup = BeautifulSoup(html_static, "html.parser")
            data_compra = self._extract_date(soup)
            itens = self._extract_items_sp(soup, data_compra)
            total_nota = self._extract_total(soup)

            # se o HTML estático já tem os sinais, retorna
            if itens or self._looks_like_danfe(soup):
                out = {
                    "tipo_documento": "gasto",
                    "itens": itens,
                    "total_nota": total_nota,
                    "data_compra": data_compra,
                    "origem": "nfce_sp_qrcode_static",
                }
                if self.enable_debug:
                    out["debug"] = self._debug_block(html_static, soup, itens)
                return out

        except Exception as e:
            logger.warning(f"Falha no requests: {e}")

        # 2) fallback Playwright (quando requests não resolve) [web:121]
        try:
            html = await self.fetch_html_dynamic(url)
            soup = BeautifulSoup(html, "html.parser")
            data_compra = self._extract_date(soup)
            itens = self._extract_items_sp(soup, data_compra)
            total_nota = self._extract_total(soup)

            out = {
                "tipo_documento": "gasto",
                "itens": itens,
                "total_nota": total_nota,
                "data_compra": data_compra,
                "origem": "nfce_sp_qrcode_browser",
            }
            if self.enable_debug:
                out["debug"] = self._debug_block(html, soup, itens)
            return out

        except ModuleNotFoundError:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "total_nota": None,
                "data_compra": None,
                "origem": "nfce_sp_qrcode",
                "mensagem": "Playwright não instalado no ambiente.",
                "debug": {"static_tried": bool(html_static)},
            }
        except Exception as e:
            return {
                "tipo_documento": "erro",
                "itens": [],
                "total_nota": None,
                "data_compra": None,
                "origem": "nfce_sp_qrcode",
                "mensagem": f"Falha ao consultar via Playwright: {str(e)}",
                "debug": {"static_tried": bool(html_static)},
            }

    # ---------------- helpers ----------------
    def _looks_like_danfe(self, soup: BeautifulSoup) -> bool:
        t = soup.get_text(" ", strip=True)
        return ("DOCUMENTO AUXILIAR" in t) or ("Chave de acesso" in t)

    def _debug_block(self, html: str, soup: BeautifulSoup, itens: List[Dict]) -> Dict:
        page_text = soup.get_text(" ", strip=True)
        return {
            "html_len": len(html or ""),
            "text_len": len(page_text),
            "text_head": page_text[:250],
            "has_codigo": "(Código:" in page_text,
            "has_qtde": "Qtde" in page_text,
            "has_vl_total": "Vl. Total" in page_text,
            "has_doc_aux": "DOCUMENTO AUXILIAR" in page_text,
            "items_found": len(itens),
        }

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