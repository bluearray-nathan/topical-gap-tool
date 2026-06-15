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

# Transient HTTP statuses worth retrying with back-off (overload, rate limits).
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_HTTP_TRIES = 3
_BACKOFF_BASE = 0.8


def _redact(text):
    """Strip any API key from text before it reaches a warning or log."""
    s = re.sub(r"(key=)[A-Za-z0-9_\-]+", r"\1REDACTED", str(text))
    k = config.get_gemini_key()
    if k:
        s = s.replace(k, "REDACTED")
    return s


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
    """Generate fan-out queries via Gemini. Retries transient errors (429/5xx) with
    back-off. The API key is sent as a header so it never appears in a URL or log."""
    queries = []
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.gemini_model()}:generateContent"
    )
    headers = {"x-goog-api-key": config.get_gemini_key() or ""}
    for _ in range(attempts):
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": temp, "candidateCount": 1},
            "systemInstruction": {"parts": [{"text": _system_instruction(country)}]},
        }
        response = None
        last_err = None
        for attempt_i in range(_HTTP_TRIES):
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_s)
                if resp.status_code in _TRANSIENT_STATUS:
                    last_err = f"{resp.status_code} {resp.reason} (transient, retrying)"
                    time.sleep(_BACKOFF_BASE * (attempt_i + 1))
                    continue
                resp.raise_for_status()
                response = resp
                break
            except ReadTimeout:
                last_err = "read timeout"
                time.sleep(_BACKOFF_BASE * (attempt_i + 1))
            except Exception as e:
                last_err = _redact(e)
                break
        if response is None:
            if last_err:
                st.warning(f"Fan-out fetch failed for '{text}': {last_err}")
            continue
        try:
            queries.extend(_parse_queries(response.json(), text))
        except Exception as e:
            st.warning(f"Error parsing fan-out response JSON for '{text}': {_redact(e)}")
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
        f'Respond with JSON only, in the form {{"queries": ["query one", "query two"]}}.'
    )


def _extract_queries(raw):
    """Pull query strings from a model response that may be a bare JSON array, an
    object with a 'queries' list, or text containing one of those."""
    if not raw:
        return []
    blocks = []
    try:
        blocks.append(json.loads(raw))
    except Exception:
        for pattern in (r"\{.*\}", r"\[.*\]"):
            m = re.search(pattern, raw, flags=re.DOTALL)
            if m:
                try:
                    blocks.append(json.loads(m.group(0)))
                except Exception:
                    pass
    for data in blocks:
        if isinstance(data, list):
            out = [q.strip() for q in data if isinstance(q, str) and q.strip()]
            if out:
                return out
        if isinstance(data, dict):
            for key in ("queries", "fan_out", "fanout", "results", "items"):
                val = data.get(key)
                if isinstance(val, list):
                    out = [q.strip() for q in val if isinstance(q, str) and q.strip()]
                    if out:
                        return out
    return []


def openai_fanout(text, country="United Kingdom", n=None):
    """ChatGPT fan-out via chat completions (model-generated query expansion)."""
    n = n or config.OPENAI_FANOUT_N
    prompt = _openai_fanout_prompt(text, country, n)
    try:
        resp = openai.chat.completions.create(
            model=config.openai_model(),
            messages=[
                {"role": "system", "content": "You output only JSON, no prose."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.warning(f"ChatGPT fan-out call failed: {e}")
        return []
    queries = _extract_queries(raw)
    if not queries:
        st.warning(f"ChatGPT fan-out returned no parseable queries. Raw response: {raw[:300]}")
    return queries


# =========================
# DISPATCHER (one or both engines)
# =========================

def _normalise(q):
    if config.NORMALIZE_YEAR_SUFFIX:
        q = text_utils.strip_trailing_years(q)
    return text_utils.strip_geo_tokens(q).strip()


def _gemini_layers(seed, country, depth, max_workers):
    """Grounded fan-out to `depth` layers, deduping between layers so the tree
    does not explode. Returns normalised queries in breadth-first order."""
    seen, ordered = set(), []

    def take(qs):
        added = []
        for q in qs or []:
            nq = _normalise(q)
            if nq and nq.lower() not in seen:
                seen.add(nq.lower())
                ordered.append(nq)
                added.append(nq)
        return added

    current = take(cached_fanouts(
        seed, config.CANDIDATE_COUNT, config.GEMINI_TEMP, config.LVL1_TIMEOUT, country
    ))
    for _ in range(max(0, depth - 1)):
        if not current:
            break
        expanded = expand_level2_parallel(
            current, attempts=config.ATTEMPTS, temp=config.GEMINI_TEMP, max_workers=max_workers,
            cand_count=config.LVL2_CANDIDATES, timeout_s=config.LVL2_TIMEOUT, country=country,
        )
        current = take(expanded)
    return ordered


def _openai_layer(seed, country, depth):
    """ChatGPT fan-out. Depth scales how many queries we ask for in the single
    call, rather than literal layers (its deeper layers would be invented on invented)."""
    seen, ordered = set(), []
    n = config.OPENAI_FANOUT_PER_DEPTH * max(1, depth)
    for q in openai_fanout(seed, country, n=n):
        nq = _normalise(q)
        if nq and nq.lower() not in seen:
            seen.add(nq.lower())
            ordered.append(nq)
    return ordered


def generate_fanout(seed, country, providers, depth=None, max_workers=None):
    """Run the selected engines to `depth`, interleave them so neither dominates
    the later cap, and merge.

    Returns (queries, source_map, engine_counts). source_map maps a lowercased
    query to the sorted engines that produced it (so 'Both' shows overlap).
    """
    providers = providers or [GEMINI]
    depth = depth or config.FANOUT_DEPTH_DEFAULT
    max_workers = max_workers or config.MAX_WORKERS

    gem = _gemini_layers(seed, country, depth, max_workers) if GEMINI in providers else []
    oai = _openai_layer(seed, country, depth) if OPENAI in providers else []
    engine_counts = {GEMINI: len(gem), OPENAI: len(oai)}

    merged, order = {}, []

    def add(q, src):
        key = q.lower()
        if key not in merged:
            merged[key] = {"display": q, "sources": set()}
            order.append(key)
        merged[key]["sources"].add(src)

    # Interleave the two engines so the later cap takes an even mix.
    i = j = 0
    while i < len(gem) or j < len(oai):
        if i < len(gem):
            add(gem[i], GEMINI)
            i += 1
        if j < len(oai):
            add(oai[j], OPENAI)
            j += 1

    queries = [merged[k]["display"] for k in order]
    source_map = {k: sorted(v["sources"]) for k, v in merged.items()}
    return queries, source_map, engine_counts
