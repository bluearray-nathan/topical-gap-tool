"""Page fetching and main-content extraction.

Improves on the original "grab every <p> and <li>" approach by isolating the
main article with trafilatura (boilerplate, nav and footer removed) and capturing
page structure: heading tree, FAQ questions and schema.org entities. No body cap.
"""

import json
import re
from dataclasses import dataclass, field

import streamlit as st
from bs4 import BeautifulSoup

import config

try:
    import trafilatura
    HAVE_TRAFILATURA = True
except Exception:
    HAVE_TRAFILATURA = False

# Schema.org @types that are structural rather than topical; excluded from the
# "declared entities" corroboration set.
_STRUCTURAL_SCHEMA = {
    "webpage", "website", "breadcrumblist", "listitem", "question", "answer",
    "imageobject", "sitenavigationelement", "searchaction", "collectionpage",
    "webpageelement", "readaction",
}


@dataclass
class PageContent:
    url: str
    h1: str = ""
    headings: list = field(default_factory=list)        # list[(level, text)]
    body: str = ""                                       # clean main-content text, uncapped
    faqs: list = field(default_factory=list)             # list[str] of questions
    jsonld_entities: list = field(default_factory=list)  # list[{type, name}]
    word_count: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.h1 or self.body)

    def headings_blob(self) -> str:
        return " | ".join(f"{lvl}:{txt}" for lvl, txt in self.headings)


# =========================
# FETCHING (fast path, then browser)
# =========================

def _fetch_fast(url, timeout=20):
    import cloudscraper
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    scraper.headers.update({"User-Agent": config.get_user_agent()})
    r = scraper.get(url, timeout=timeout, proxies=config.get_requests_proxy_map())
    r.raise_for_status()
    return r.text


def _fetch_playwright(url):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, proxy=config.get_playwright_proxy_settings()
        )
        context = browser.new_context(user_agent=config.get_user_agent())
        page = context.new_page()
        resp = page.goto(url, timeout=45000)
        status = resp.status if resp else None
        html = page.content()
        context.close()
        browser.close()
    if status == 403:
        raise RuntimeError("HTTP 403 Forbidden")
    return html


def fetch_html(url, timeout=20):
    """Fast path via cloudscraper; fall back to a headless browser for JS pages."""
    html = None
    try:
        html = _fetch_fast(url, timeout)
    except Exception:
        html = None
    if html and re.search(r"<h1[ >]", html, re.I):
        return html
    try:
        return _fetch_playwright(url)
    except Exception:
        if html:
            return html
        raise


def _friendly_fetch_error(e) -> str:
    err = str(e)
    if "libglib" in err or "BrowserType.launch" in err or "shared object file" in err:
        return (
            "This page requires a browser to load but the browser engine is "
            "unavailable. Try re-deploying the app; if it persists, contact support."
        )
    return f"Fetch error ({e})"


# =========================
# SCHEMA / JSON-LD PARSING
# =========================

def _iter_jsonld_nodes(data):
    if isinstance(data, list):
        for item in data:
            yield from _iter_jsonld_nodes(item)
    elif isinstance(data, dict):
        if isinstance(data.get("@graph"), list):
            for item in data["@graph"]:
                yield from _iter_jsonld_nodes(item)
        yield data
        # FAQ questions live in mainEntity
        main = data.get("mainEntity")
        if isinstance(main, (list, dict)):
            yield from _iter_jsonld_nodes(main)


def _parse_jsonld(soup):
    nodes = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for node in _iter_jsonld_nodes(data):
            if not isinstance(node, dict):
                continue
            t = node.get("@type")
            name = node.get("name")
            if isinstance(t, list):
                t = ", ".join(str(x) for x in t)
            if t and name and isinstance(name, str):
                nodes.append({"type": str(t), "name": name.strip()})
    seen, ded = set(), []
    for e in nodes:
        key = (e["type"].lower(), e["name"].lower())
        if key not in seen:
            seen.add(key)
            ded.append(e)
    return ded


def _split_schema(nodes):
    """Return (declared_entities, faq_questions) from parsed JSON-LD nodes."""
    entities, questions = [], []
    for e in nodes:
        t = e["type"].lower()
        if "question" in t:
            questions.append(e["name"])
        elif not any(s in t for s in _STRUCTURAL_SCHEMA):
            entities.append(e)
    return entities, questions


# =========================
# BODY + STRUCTURE
# =========================

def _extract_body(html, url, struct_soup):
    if HAVE_TRAFILATURA:
        try:
            txt = trafilatura.extract(
                html,
                include_tables=True,
                include_comments=False,
                include_formatting=False,
                favor_recall=True,
                url=url,
            )
            if txt and len(txt.split()) >= 40:
                return txt.strip()
        except Exception:
            pass
    # Fallback: paragraphs + list items from the de-boilerplated soup.
    parts = [p.get_text(strip=True) for p in struct_soup.find_all("p")]
    parts += [li.get_text(strip=True) for li in struct_soup.find_all("li")]
    return "\n".join(t for t in parts if t)


def _collect_faqs(struct_soup, headings, schema_questions):
    questions = list(schema_questions)
    for _lvl, txt in headings:
        if txt.strip().endswith("?"):
            questions.append(txt.strip())
    for s in struct_soup.find_all("summary"):
        t = s.get_text(strip=True)
        if t:
            questions.append(t)
    seen, ded = set(), []
    for q in questions:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            ded.append(q)
    return ded


def parse_html(url, html) -> PageContent:
    """Parse fetched HTML into structured, boilerplate-free content.

    Separated from fetching so it can be tested without network access.
    """
    if not html:
        return PageContent(url=url, error="Empty response from page")

    soup = BeautifulSoup(html, "html.parser")
    schema_nodes = _parse_jsonld(soup)
    declared_entities, schema_questions = _split_schema(schema_nodes)

    # Structural copy with chrome removed for heading/FAQ extraction.
    struct = BeautifulSoup(html, "html.parser")
    for tag in struct.find_all(["nav", "footer", "aside", "form", "script", "style", "noscript"]):
        tag.decompose()
    for tag in struct.select('[role="navigation"], [role="contentinfo"]'):
        tag.decompose()

    h1_tag = struct.find("h1")
    h1 = h1_tag.get_text(strip=True) if h1_tag else ""
    headings = [
        (t.name.upper(), t.get_text(strip=True))
        for t in struct.find_all(["h2", "h3", "h4"])
        if t.get_text(strip=True)
    ]

    body = _extract_body(html, url, struct)
    faqs = _collect_faqs(struct, headings, schema_questions)

    return PageContent(
        url=url,
        h1=h1,
        headings=headings,
        body=body,
        faqs=faqs,
        jsonld_entities=declared_entities,
        word_count=len(body.split()),
    )


def extract_page(url) -> PageContent:
    """Fetch and parse a URL into structured, boilerplate-free content."""
    try:
        html = fetch_html(url)
    except Exception as e:
        return PageContent(url=url, error=_friendly_fetch_error(e))
    return parse_html(url, html)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_extract_page(url) -> PageContent:
    return extract_page(url)
