"""Fan-out query generation via Gemini with Google Search grounding.

Behaviour matches the original implementation. The model name is read from
config so a model rename does not silently break generation.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import requests
import streamlit as st
from requests.exceptions import ReadTimeout

import config
import text_utils

GEMINI = "Gemini"
OPENAI = "ChatGPT"


def _system_instruction(country):
    return (
        f"Your task: use the google_search tool to discover related search queries "
        f"that real users in {country} would type when researching this topic. "
        f"You MUST call google_search; do NOT answer the input directly from your own knowledge, "
        f"do NOT write guides, explanations, or prose responses. "
        f"Only perform searches so the grounding metadata contains the fan-out queries. "
        f"Return between 4 and 6 broad, distinct queries that each cover a different angle of the topic. "
        f"Do NOT return highly granular variations that only differ by brand/model name "
        f"(e.g. avoid separate queries for every EV manufacturer when one generic query covers the same intent). "
        f"Prefer broader searches that represent content gaps at the section level, not the bullet level. "
        f"Treat {country} as the IMPLICIT location context for each search; do NOT include "
        f"'{country}', country abbreviations (e.g. 'UK', 'US', 'USA', 'GB'), city names, "
        f"or region names as literal keywords inside the search queries. "
        f"Real users searching locally do not type their country in the query; Google infers location. "
        f"Each search should read like a natural query a local resident would type, "
        f"using local language/spelling, referencing {country} regulations, pricing in local currency, "
        f"and {country} providers/brands/suppliers where specific brand/product searches are needed. "
        f"Do NOT search for topics, legislation, brands, or regulatory bodies specific to other countries. "
        f"If the topic has multiple geographic angles, search only the {country} angle."
    )


def _parse_queries(resp_json, text):
    queries, found_any = [], False
    for cand in resp_json.get("candidates", []):
        # Path 1 (Gemini 2.x): groundingMetadata.webSearchQueries
        gm = cand.get("groundingMetadata") or cand.get("grounding_metadata") or {}
        web_qs = (
            gm.get("webSearchQueries")
            or gm.get("web_search_queries")
            or gm.get("searchQueries")
            or []
        )
        if web_qs:
            found_any = True
            queries.extend(web_qs)
        # Path 2 (Gemini 3): functionCall.args.queries
        for part in cand.get("content", {}).get("parts", []) or []:
            fc = part.get("functionCall") or part.get("function_call") or {}
            if fc.get("name") == "google_search":
                fc_qs = (fc.get("args") or {}).get("queries") or []
                if fc_qs:
                    found_any = True
                    queries.extend(fc_qs)
    if not found_any:
        preview = json.dumps(resp_json, indent=2)[:1500]
        st.warning(f"No grounding queries in response for '{text}'. Raw preview:\n{preview}")
    return queries


def fetch_query_fan_outs_multi(text, attempts=1, temp=0.0, cand_count=None, timeout_s=60,
                               country="United Kingdom"):
    """Generate fan-out queries via Gemini, retrying on ReadTimeout."""
    queries = []
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.gemini_model()}:generateContent?key={config.get_gemini_key()}"
    )
    for _ in range(attempts):
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": temp, "candidateCount": 1},
            "systemInstruction": {"parts": [{"text": _system_instruction(country)}]},
        }
        response = None
        for _retry in range(2):
            try:
                response = requests.post(endpoint, json=payload, timeout=timeout_s)
                response.raise_for_status()
                break
            except ReadTimeout:
                time.sleep(0.8)
            except Exception as e:
                body = ""
                if response is not None:
                    try:
                        body = response.text[:500]
                    except Exception:
                        pass
                st.warning(f"Fan-out fetch failed for '{text}': {e}{' — ' + body if body else ''}")
                break
        if not response:
            continue
        try:
            queries.extend(_parse_queries(response.json(), text))
        except Exception as e:
            st.warning(f"Error parsing fan-out response JSON for '{text}': {e}")
    return queries


def expand_level2_parallel(level1, attempts=1, temp=0.4, max_workers=6, cand_count=4,
                           timeout_s=15, country="United Kingdom"):
    results = []
    if not level1:
        return results
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(level1)))) as ex:
        futs = {
            ex.submit(fetch_query_fan_outs_multi, q, attempts, temp, cand_count, timeout_s, country): q
            for q in level1
        }
        for fut in as_completed(futs):
            try:
                results.extend(fut.result() or [])
            except Exception as e:
                st.warning(f"Level-2 expansion error: {e}")
    return results


@st.cache_data(show_spinner=False, ttl=86400)
def cached_fanouts(text, cand_count, temp, timeout_s, country="United Kingdom"):
    return fetch_query_fan_outs_multi(
        text, attempts=config.ATTEMPTS, temp=temp, cand_count=cand_count,
        timeout_s=timeout_s, country=country,
    )


# =========================
# OPENAI (CHATGPT) FAN-OUT
# =========================

def _openai_fanout_prompt(text, country, n):
    return (
        f"You are modelling how an AI search engine fans a topic out into the distinct search "
        f"queries a real user in {country} would issue while researching it.\n"
        f"Topic: {text}\n\n"
        f"Return {n} broad, distinct queries that each cover a different angle or sub-topic. "
        f"Prefer section-level intents over trivial variations. Do not include the country name "
        f"or its abbreviations in the queries; assume local context. Use local spelling, currency, "
        f"brands and regulations where relevant.\n"
        f"Return ONLY a JSON array of strings."
    )


def _parse_json_array(text):
    if not text:
        return []
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    return [q.strip() for q in arr if isinstance(q, str) and q.strip()]


def _openai_fanout_grounded(prompt):
    """Best effort: use the Responses API web_search tool, extract issued queries
    and any JSON array the model returns. Returns [] if the API is unavailable."""
    try:
        resp = openai.responses.create(
            model=config.openai_model(),
            tools=[{"type": "web_search"}],
            input=prompt + "\n\nSearch the web first, then return the JSON array.",
        )
    except Exception:
        return []
    out = []
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "web_search_call":
                action = getattr(item, "action", None)
                q = getattr(action, "query", None) if action is not None else None
                if q:
                    out.append(q)
    except Exception:
        pass
    out.extend(_parse_json_array(getattr(resp, "output_text", "") or ""))
    seen, ded = set(), []
    for q in out:
        k = q.strip().lower()
        if q.strip() and k not in seen:
            seen.add(k)
            ded.append(q.strip())
    return ded


def _openai_fanout_plain(prompt):
    try:
        resp = openai.chat.completions.create(
            model=config.openai_model(),
            messages=[
                {"role": "system", "content": "You output ONLY a JSON array of strings."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.OPENAI_FANOUT_TEMP,
        )
        return _parse_json_array((resp.choices[0].message.content or "").strip())
    except Exception as e:
        st.warning(f"ChatGPT fan-out failed: {e}")
        return []


def openai_fanout(text, country="United Kingdom", n=None):
    """ChatGPT fan-out: grounded via web search where supported, else generated."""
    n = n or config.OPENAI_FANOUT_N
    prompt = _openai_fanout_prompt(text, country, n)
    queries = _openai_fanout_grounded(prompt)
    if queries:
        return queries
    return _openai_fanout_plain(prompt)


@st.cache_data(show_spinner=False, ttl=86400)
def cached_openai_fanout(text, country="United Kingdom", n=None):
    return openai_fanout(text, country=country, n=n)


# =========================
# DISPATCHER (one or both engines)
# =========================

def _normalise(q):
    if config.NORMALIZE_YEAR_SUFFIX:
        q = text_utils.strip_trailing_years(q)
    return text_utils.strip_geo_tokens(q).strip()


def generate_fanout(seed, country, providers, max_workers=None):
    """Run the selected fan-out engines and merge.

    Returns (queries, source_map) where source_map maps a lowercased query to the
    sorted list of engines that produced it (so 'Both' shows overlap).
    """
    providers = providers or [GEMINI]
    max_workers = max_workers or config.MAX_WORKERS
    merged = {}  # lower -> {"display":.., "sources": set()}

    def add(q, src):
        q = _normalise(q)
        if not q:
            return
        key = q.lower()
        entry = merged.setdefault(key, {"display": q, "sources": set()})
        entry["sources"].add(src)

    if GEMINI in providers:
        lvl1 = cached_fanouts(seed, config.CANDIDATE_COUNT, config.GEMINI_TEMP, config.LVL1_TIMEOUT, country)
        lvl2 = expand_level2_parallel(
            lvl1, attempts=config.ATTEMPTS, temp=config.GEMINI_TEMP, max_workers=max_workers,
            cand_count=config.LVL2_CANDIDATES, timeout_s=config.LVL2_TIMEOUT, country=country,
        )
        for q in (lvl1 or []) + (lvl2 or []):
            add(q, GEMINI)

    if OPENAI in providers:
        for q in cached_openai_fanout(seed, country):
            add(q, OPENAI)

    queries = [v["display"] for v in merged.values()]
    source_map = {k: sorted(v["sources"]) for k, v in merged.items()}
    return queries, source_map
