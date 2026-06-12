"""Competitor benchmarking.

Discovers competitors from the live SERP (SerpAPI) with a manual-paste fallback,
fetches and scores each one against the page's entity set, and builds the
comparison views: an entity-by-site matrix and a competitor advantage list.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
import streamlit as st

import config
import coverage
import extraction


def serpapi_available() -> bool:
    return bool(config.get_serpapi_key())


def domain_of(url) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return url


def discover_competitors(keyword, country, own_url="", limit=None):
    """Top organic results for ``keyword`` in ``country`` via SerpAPI, own domain
    removed and one result per domain. Returns [] when no SerpAPI key is set."""
    limit = limit or config.MAX_COMPETITORS
    key = config.get_serpapi_key()
    if not key or not keyword:
        return []
    params = {
        "engine": "google",
        "q": keyword,
        "gl": config.SERP_COUNTRY_CODE.get(country, "us"),
        "hl": "en",
        "num": 10,
        "api_key": key,
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.warning(f"Competitor discovery failed: {e}")
        return []

    own_domain = domain_of(own_url) if own_url else ""
    urls, seen = [], set()
    for item in data.get("organic_results", []) or []:
        link = item.get("link")
        if not link:
            continue
        d = domain_of(link)
        if (own_domain and d == own_domain) or d in seen:
            continue
        seen.add(d)
        urls.append(link)
        if len(urls) >= limit:
            break
    return urls


def _entity_items(entities):
    return [
        {"name": e["name"], "queries": e["queries"], "definition": e["definition"]}
        for e in entities
    ]


def _score_one(url, items):
    page = extraction.cached_extract_page(url, need_h1=False)
    if not page.ok:
        return url, {"ok": False, "error": page.error or "No usable content", "scores": {}}
    results = coverage.score_coverage(page, items)
    return url, {
        "ok": True,
        "error": None,
        "title": page.h1,
        "word_count": page.word_count,
        "scores": {r["label"]: r for r in results},
    }


def benchmark_competitors(entities, competitor_urls, max_workers=None):
    """Fetch and score each competitor against the page's entity set, in parallel."""
    out = {}
    if not entities or not competitor_urls:
        return out
    items = _entity_items(entities)
    max_workers = max_workers or config.MAX_WORKERS
    with ThreadPoolExecutor(max_workers=min(max_workers, len(competitor_urls))) as ex:
        futs = {ex.submit(_score_one, u, items): u for u in competitor_urls}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                url, res = fut.result()
                out[url] = res
            except Exception as e:
                out[u] = {"ok": False, "error": str(e), "scores": {}}
    return out


def build_matrix(entity_rows, competitor_results):
    """Entity-by-site status matrix. Returns (rows, competitor_label_map)."""
    comp_urls = list(competitor_results.keys())
    labels = {u: domain_of(u) for u in comp_urls}
    rows = []
    for er in entity_rows:
        name = er["entity"]
        row = {"Entity": name, "Type": er["type"], "Your page": er["status"]}
        for u in comp_urls:
            res = competitor_results[u]
            if not res.get("ok"):
                row[labels[u]] = "n/a"
            else:
                cell = res["scores"].get(name)
                row[labels[u]] = cell["status"] if cell else "Missing"
        rows.append(row)
    return rows, labels


def competitor_advantage(entity_rows, competitor_results):
    """Entities where your page is missing or thin but a rival covers them well."""
    comp_urls = list(competitor_results.keys())
    rows = []
    for er in entity_rows:
        if er["status"] == "Strong":
            continue
        name = er["entity"]
        covering = []
        for u in comp_urls:
            res = competitor_results[u]
            if not res.get("ok"):
                continue
            cell = res["scores"].get(name)
            if cell and cell["score"] >= 2:
                covering.append(domain_of(u))
        if covering:
            rows.append({
                "Entity": name,
                "Type": er["type"],
                "Your page": er["status"],
                "Competitors covering well": len(covering),
                "Which competitors": ", ".join(covering),
                "Queries": "; ".join(er["queries"]),
            })
    rows.sort(key=lambda x: x["Competitors covering well"], reverse=True)
    return rows
