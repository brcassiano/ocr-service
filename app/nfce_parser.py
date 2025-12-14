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
                logger.info(f"[PW] url_final={page.url}")
                logger.info(f"[PW] html_len={len(html)}")

                # salva arquivo pra inspecionar no container (ajuda demais)
                with open("/tmp/nfce_sp.html", "w", encoding="utf-8") as f:
                    f.write(html)

                # e salva também um “texto limpo” (pra validar regex)
                txt = await page.locator("body").inner_text()
                with open("/tmp/nfce_sp.txt", "w", encoding="utf-8") as f:
                    f.write(txt)

                logger.info(f"[PW] body_text_len={len(txt)}")
                logger.info(f"[PW] body_text_head={txt[:300]}")

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
        text = soup.get_text(" ", strip=True)

        # normalizações
        text = text.replace("**", " ")
        text = text.replace("|", " ")
        text = re.sub(r"\s+", " ", text).strip()

        itens: List[Dict] = []

        item_re = re.compile(
            r"(?P<desc>.+?)\s*\(Código:\s*(?P<codigo>[^)]+)\)\s*"
            r".*?Qtde\.\:?\s*(?P<qtd>[0-9,.]+)\s*"
            r".*?UN\:\s*(?P<un>[A-Z]+)\s*"
            r".*?Vl\.\s*Unit\.\:?\s*(?P<vu>[0-9,.]+)\s*"
            r".*?Vl\.\s*Total\s*(?P<vt>[0-9,.]+)",
            re.IGNORECASE,
        ),

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
                },
            

            debug = {
                    "html_len": len(html or ""),
                    "text_head": soup.get_text(" ", strip=True)[:250],
                    "match_codigo": "(Código:" in soup.get_text(" ", strip=True),
                    "match_vl_total": "Vl. Total" in soup.get_text(" ", strip=True),
                    "items_found": len(itens),
                }

            )

        return {
            "tipo_documento": "gasto",
            "itens": itens,
            "total_nota": total_nota,
            "data_compra": data_compra,
            "origem": "nfce_sp_qrcode_browser",
            "debug": debug,
        }


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
