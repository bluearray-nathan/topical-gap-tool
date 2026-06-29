"""Page fetching and main-content extraction.

Improves on the original "grab every <p> and <li>" approach by isolating the
main article with trafilatura (boilerplate, nav and footer removed) and capturing
page structure: heading tree, FAQ questions and schema.org entities. No body cap.
"""

import json
import re
from dataclasses import dataclass, field

import requests
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
    title: str = ""                                      # <title>, used as a seed fallback
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


def fetch_html(url, timeout=20, need_h1=True):
    """Fast path via cloudscraper; fall back to a headless browser when needed.

    When need_h1 is False (e.g. competitor pages) we accept whatever the fast path
    returns rather than forcing the browser, so competitor analysis keeps working
    even where the browser engine is unavailable.
    """
    try:
        html = _fetch_fast(url, timeout)
    except Exception:
        html = None
    if html and (not need_h1 or re.search(r"<h1[ >]", html, re.I)):
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


MIN_CONTENT_WORDS = 50

# Markers that the response is an anti-bot challenge or interstitial rather than the
# real page (Cloudflare and similar). When present, the content is not usable, and
# auditing it would fan out from a meaningless seed.
_BLOCK_MARKERS = (
    "just a moment...",
    "attention required! | cloudflare",
    "cf-browser-verification",
    "cf_chl_opt",
    "challenge-platform",
    "checking your browser before accessing",
    "enable javascript and cookies to continue",
    "ddos protection by cloudflare",
    "please turn javascript on and reload the page",
)


def _looks_blocked(html) -> bool:
    head = html[:8000].lower()
    return any(m in head for m in _BLOCK_MARKERS)


def parse_html(url, html) -> PageContent:
    """Parse fetched HTML into structured, boilerplate-free content.

    Separated from fetching so it can be tested without network access. Sets an error
    (so the caller skips the page) when the response is an anti-bot challenge or has too
    little real content to analyse, rather than letting the audit fan out from nothing.
    """
    if not html:
        return PageContent(url=url, error="Empty response from page")

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    if _looks_blocked(html):
        return PageContent(
            url=url, title=title,
            error="Blocked by anti-bot protection (likely Cloudflare). Could not read the page "
                  "content. This site needs a proxy or a working browser fetch.",
        )

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
    word_count = len(body.split())

    if word_count < MIN_CONTENT_WORDS and not h1:
        return PageContent(
            url=url, title=title, h1=h1, headings=headings, body=body, word_count=word_count,
            error="Could not read enough page content to analyse. The page may be "
                  "JavaScript-rendered or blocked by anti-bot protection.",
        )

    return PageContent(
        url=url,
        h1=h1,
        title=title,
        headings=headings,
        body=body,
        faqs=faqs,
        jsonld_entities=declared_entities,
        word_count=word_count,
    )


def fetch_via_dataforseo(url, timeout=60):
    """Fetch a page's raw HTML through DataForSEO's On-Page API, for pages the direct
    crawler cannot read (e.g. Cloudflare). Returns HTML or None.

    Two calls: instant_pages (JS rendering + store_raw_html) fetches and stores the page,
    then raw_html retrieves the stored markup. Authenticated with the DataForSEO login
    and password from secrets.
    """
    login = config.get_dataforseo_login()
    password = config.get_dataforseo_password()
    if not (login and password):
        return None
    auth = (login, password)
    try:
        r1 = requests.post(
            "https://api.dataforseo.com/v3/on_page/instant_pages",
            auth=auth, timeout=timeout,
            json=[{"url": url, "enable_javascript": True, "store_raw_html": True,
                   "custom_user_agent": config.get_user_agent()}],
        )
        r1.raise_for_status()
        task = (r1.json().get("tasks") or [{}])[0]
        task_id = task.get("id")
        if not task_id:
            st.warning(f"DataForSEO fetch failed for {url}: {task.get('status_message', 'no task id')}")
            return None

        r2 = requests.post(
            "https://api.dataforseo.com/v3/on_page/raw_html",
            auth=auth, timeout=timeout,
            json=[{"id": task_id, "url": url}],
        )
        r2.raise_for_status()
        result = ((r2.json().get("tasks") or [{}])[0].get("result") or [{}])[0] or {}
        items = result.get("items")
        if isinstance(items, list):
            return (items[0] or {}).get("html") if items else None
        if isinstance(items, dict):
            return items.get("html")
        return None
    except Exception as e:
        st.warning(f"DataForSEO fetch failed for {url}: {e}")
        return None


def extract_page(url, need_h1=True) -> PageContent:
    """Fetch and parse a URL into structured, boilerplate-free content.

    Falls back to DataForSEO (when configured) if the direct fetch is blocked or
    unreadable, so anti-bot-protected pages can still be analysed.
    """
    try:
        html = fetch_html(url, need_h1=need_h1)
        page = parse_html(url, html)
    except Exception as e:
        page = PageContent(url=url, error=_friendly_fetch_error(e))

    if not page.ok and config.dataforseo_available():
        df_html = fetch_via_dataforseo(url)
        if df_html:
            df_page = parse_html(url, df_html)
            if df_page.ok:
                return df_page
    return page


@st.cache_data(show_spinner=False, ttl=3600)
def cached_extract_page(url, need_h1=True) -> PageContent:
    return extract_page(url, need_h1=need_h1)
