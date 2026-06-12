"""Fan-out query generation via Gemini with Google Search grounding.

Behaviour matches the original implementation. The model name is read from
config so a model rename does not silently break generation.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
from requests.exceptions import ReadTimeout

import config


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
