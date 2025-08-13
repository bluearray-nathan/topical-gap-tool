import subprocess
import sys
# Ensure Playwright is installed for browser automation
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)

import time
import json
import re
import requests
import numpy as np
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import cloudscraper
import openai
from requests.exceptions import ReadTimeout
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Optional fuzzy matching (RapidFuzz). Falls back to a simple Jaccard if unavailable.
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Overview/AI Mode query fan-out gap analysis", layout="wide")
st.title("🔍 AI Overview/AI Mode Query Fan-Out Gap Analysis")

st.write(
    """
This tool identifies content gaps by:
1) Extracting your page's H1, headings, and body text.  
2) Generating user query fan-outs for the H1 (Gemini).  
3) Doing a second-level fan-out on those queries.  
4) Comparing all queries against the page content to find missing topics.
"""
)

# --- Fixed Configuration (all defaults, no UI controls) ---
gemini_temp       = 0.4      # Diversity for fan-out generation
gpt_temp          = 0.1      # Temperature for gap reasoning
attempts          = 1        # Gemini calls per input
candidate_count   = 7        # Default candidates per Gemini call
BODY_CHAR_LIMIT   = 2000     # Limit body text passed to GPT per batch

# Dedupe defaults
ENABLE_DEDUPE     = True
FUZZY_RATIO       = 92
EMBED_ON          = True
EMBED_THRESHOLD   = 0.86
EMBED_MODEL       = "text-embedding-3-small"

# Performance defaults
MAX_WORKERS       = 6
LVL2_CANDIDATES   = 4
LVL2_TIMEOUT      = 15
LVL1_TIMEOUT      = 30
GPT_BATCH_SIZE    = 8

# Normalization default
NORMALIZE_YEAR_SUFFIX = True

# --- Load API Keys from Streamlit Secrets ---
openai.api_key   = st.secrets["openai"]["api_key"]
gemini_api_key   = st.secrets["google"]["gemini_api_key"]

# --- URL Input Area ---
urls_input = st.text_area(
    "Enter one URL per line to audit:",
    placeholder="https://example.com/page1\nhttps://example.com/page2"
)

# --- Style the Start Button ---
st.markdown(
    """
    <style>
    div.stButton > button:first-child { background-color: #e63946; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize Session State ---
def init_state():
    defaults = {
        "last_urls": [],
        "processed": False,
        "detailed": [],
        "summary": [],
        "actions": [],
        "skipped": [],
        "h1_fanout_cache": {},
        "fanout_layer2_cache": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
init_state()

# --- Parse and Detect URL List ---
urls = [line.strip() for line in urls_input.splitlines() if line.strip()]

# --- Reset state when URLs change ---
if urls and st.session_state.last_urls != urls:
    st.session_state.last_urls = urls
    st.session_state.processed = False
    st.session_state.detailed.clear()
    st.session_state.summary.clear()
    st.session_state.actions.clear()
    st.session_state.skipped.clear()
    st.session_state.h1_fanout_cache.clear()
    st.session_state.fanout_layer2_cache.clear()

# =========================
# NORMALIZATION UTILITIES
# =========================

# Handles endings like: " ... 2024", " ... (2024)", " ... in 2024", " ... 2024/25", " ... 2024-2025"
YEAR_SUFFIX_RE = re.compile(r"""
    (?:\s*[\(\-]?\s*)        # optional spacing / '(' / '-' before the year
    (?:in\s+)?               # optional 'in '
    ((?:19|20)\d{2})         # base 4-digit year (capture)
    (?:\s*/\s*\d{2}          # ' /25' short range
       |-(?:19|20)\d{2}      # or '-2025' full range
    )?                       # optional year-range
    \)?\s*$                  # optional ')' then end of string
""", re.IGNORECASE | re.VERBOSE)

def strip_trailing_years(q: str, min_year: int = 2000, max_year: int | None = None) -> str:
    """Remove a trailing year or year-range only at the end of the query."""
    if max_year is None:
        max_year = datetime.now().year + 1  # allow next-year planning queries
    m = YEAR_SUFFIX_RE.search(q)
    if not m:
        return q
    try:
        y = int(m.group(1))
    except Exception:
        return q
    if min_year <= y <= max_year:
        return q[:m.start()].rstrip()
    return q

# =========================
# DEDUPE UTILITIES
# =========================

STOPWORDS = {
    "the","and","of","in","to","a","for","with","on","about",
    "vs","vs.","is","are","your","what","how","why","more",
    "latest","new","does","do","an"
}

def canonicalize(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^\w\s]", " ", q)
    if STOPWORDS:
        q = re.sub(r"\b(" + "|".join(map(re.escape, STOPWORDS)) + r")\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _token_set_ratio(a: str, b: str) -> int:
    if HAVE_RAPIDFUZZ:
        return int(fuzz.token_set_ratio(a, b))
    ta, tb = set(canonicalize(a).split()), set(canonicalize(b).split())
    if not ta and not tb: return 100
    if not ta or not tb:  return 0
    jacc = len(ta & tb) / len(ta | tb)
    return int(round(jacc * 100))

def dedupe_exact(queries):
    seen = set()
    kept, groups = [], {}
    for q in queries:
        c = canonicalize(q)
        if c in seen:
            for rep in kept:
                if canonicalize(rep) == c:
                    groups[rep].append(q)
                    break
        else:
            seen.add(c)
            kept.append(q)
            groups[q] = [q]
    return kept, groups

def dedupe_token_set(queries, min_ratio=92):
    kept = []
    groups = {}
    for q in queries:
        attached = False
        for k in kept:
            score = _token_set_ratio(q, k)
            if score >= min_ratio:
                groups[k].append(q)
                attached = True
                break
        if not attached:
            kept.append(q)
            groups[q] = [q]
    return kept, groups

def get_embeddings(queries, model="text-embedding-3-small"):
    resp = openai.embeddings.create(model=model, input=queries)
    data = getattr(resp, "data", None) or resp



















