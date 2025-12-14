import re
import logging
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class NfceParserSP:
    """
    Parser NFC-e SP via QRCode usando HTML retornado pelo endpoint público.

    Importante: como no seu ambiente o Playwright está recebendo apenas o "menu",
    este parser foca em requests + BeautifulSoup/regex, que (pelo seu HTML) contém
    o DANFE completo com itens. [web:2]
    """

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

    def fetch_html(self, url: str) -> str:
        url_clean = self._clean_url(url)

        s = requests.Session()
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        resp = s.get(url_clean, headers=headers, timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp.text

    async def parse(self, url: str) -> Dict:
        html = self.fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

        data_compra = self._extract_date(soup)
        itens = self._extract_items_sp(soup, data_compra)
        total_nota = self._extract_total(soup)

        out = {
            "tipo_documento": "gasto",
            "itens": itens,
            "total_nota": total_nota,
            "data_compra": data_compra,
            "origem": "nfce_sp_qrcode_static",
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
            "has_doc_aux": "DOCUMENTO AUXILIAR" in page_text,
            "has_codigo": "(Código:" in page_text,
            "has_qtde": "Qtde" in page_text,
            "has_vl_total": "Vl. Total" in page_text,
            "items_found": len(itens),
        }

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        txt = soup.get_text(" ", strip=True)
        # "Emissão: 11/12/2025 18:57:55"
        m = re.search(r"Emissão:\s*(\d{2}/\d{2}/\d{4})", txt, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    def _extract_total(self, soup: BeautifulSoup) -> Optional[float]:
        txt = soup.get_text(" ", strip=True)
        # "Valor a pagar R$:236,09"
        m = re.search(r"Valor a pagar\s*R\$\:?\s*([0-9.,]+)", txt, re.IGNORECASE)
        if m:
            return self._to_float(m.group(1))
        return None

    def _extract_items_sp(self, soup: BeautifulSoup, data_compra: Optional[str]) -> List[Dict]:
        # O HTML do SP vem com markdown-like "**Qtde.:**" e pipes "|" [web:2]
        text = soup.get_text(" ", strip=True)
        text = text.replace("|", " ")
        text = re.sub(r"\s+", " ", text).strip()

        itens: List[Dict] = []

        # Aceita:
        # "... (Código: 789...) **Qtde.:**1 **UN:** UN **Vl. Unit.:** 15,9 | Vl. Total 15,90" [web:2]
        item_re = re.compile(
            r"(?P<desc>.+?)\s*\(Código:\s*(?P<codigo>[^)]+)\)\s*"
            r".*?Qtde\.?:\*{0,2}\s*(?P<qtd>[0-9,.]+)\s*"
            r".*?UN:\*{0,2}\s*(?P<un>[A-Z]{1,3})\s*"
            r".*?Vl\.\s*Unit\.?:\*{0,2}\s*(?P<vu>[0-9,.]+)\s*"
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

    def _to_float(self, s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s = str(s).strip().replace(" ", "")
        # 1.234,56 -> 1234.56
        if s.count(",") == 1 and s.count(".") >= 1:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None