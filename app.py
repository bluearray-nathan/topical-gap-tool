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
import uuid

# Optional fuzzy matching (RapidFuzz). Falls back to a simple Jaccard if unavailable.
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

# Optional Google Cloud Storage for checkpointing/resume
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    HAVE_GCS = True
except Exception:
    HAVE_GCS = False

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Overview/AI Mode query fan-out gap analysis", layout="wide")
st.title("🔍 AI Overview/AI Mode Query Fan-Out Gap Analysis")

st.sidebar.header("About this tool")
st.sidebar.write(
    """
This tool identifies content gaps by:
1) Extracting your page's H1, headings, and body text.
2) Generating user query fan-outs for the H1 (Gemini).
3) Doing a second-level fan-out on those queries.
4) Comparing all queries against the page content to find missing topics.
"""
)

# --- Fixed Configuration ---
gemini_temp       = 0.4      # Diversity for fan-out generation
gpt_temp          = 0.1      # Temperature for gap reasoning
attempts          = 1        # Gemini calls per input
candidate_count   = 7        # Default candidates per Gemini call
BODY_CHAR_LIMIT   = 2000     # Limit body text passed to GPT per batch

# --- Dedupe Controls (Sidebar) ---
st.sidebar.subheader("Dedupe Options")
enable_dedupe      = st.sidebar.checkbox("Enable dedupe", value=True)
fuzzy_ratio        = st.sidebar.slider("Fuzzy token-set threshold", 80, 100, 92)
embed_on           = st.sidebar.checkbox("Use embedding dedupe", value=True)
embed_thr_pct      = st.sidebar.slider("Embedding cosine threshold (%)", 70, 99, 86)
embed_threshold    = embed_thr_pct / 100.0
embed_model        = st.sidebar.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"], index=0)

# --- Performance Controls (Sidebar) ---
st.sidebar.subheader("Performance")
max_workers        = st.sidebar.slider("Level-2 parallel workers", 2, 12, 6)
lvl2_candidates    = st.sidebar.slider("Level-2 candidateCount", 1, 7, 4)
lvl2_timeout       = st.sidebar.slider("Level-2 timeout (seconds)", 5, 60, 15)
lvl1_timeout       = st.sidebar.slider("Level-1 timeout (seconds)", 5, 60, 30)
gpt_batch_size     = st.sidebar.slider("GPT coverage batch size", 4, 16, 8)

# --- Normalization (Sidebar) ---
st.sidebar.subheader("Normalization")
normalize_year_suffix = st.sidebar.checkbox("Strip trailing years (e.g., '2024' or '2024/25')", value=True)

# --- Cloud Checkpointing (Sidebar) ---
st.sidebar.subheader("Cloud Checkpointing (GCS)")
use_gcs = st.sidebar.checkbox("Save results to Google Cloud Storage (resume later)", value=False, help="Requires google-cloud-storage and service account credentials in st.secrets.")
# Defaults pulled from secrets if present
_default_bucket = (
    st.secrets.get("gcp", {}).get("bucket", "") if hasattr(st, "secrets") else ""
)
_default_prefix = (
    st.secrets.get("gcp", {}).get("prefix", "gap-jobs") if hasattr(st, "secrets") else "gap-jobs"
)
gcs_bucket_name = st.sidebar.text_input("GCS bucket name", _default_bucket)
gcs_prefix = st.sidebar.text_input("GCS prefix", _default_prefix)
job_id_input = st.sidebar.text_input("Job ID (leave blank to generate)", "")
resume_existing = st.sidebar.checkbox("Resume existing Job ID", value=False)

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
        "active_job_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
init_state()

# --- Parse and Detect URL List ---
urls = [line.strip() for line in urls_input.splitlines() if line.strip()]

# --- GCS Helpers ---
_gcs_client = None
_gcs_bucket = None

def _init_gcs():
    global _gcs_client, _gcs_bucket
    if not use_gcs:
        return False
    if not HAVE_GCS:
        st.error("google-cloud-storage is not installed. Add it to requirements and restart.")
        return False
    if not gcs_bucket_name:
        st.error("Please set a GCS bucket name in the sidebar.")
        return False
    try:
        # Prefer explicit service account from secrets
        sa_info = st.secrets.get("gcp_service_account", None)
        if sa_info:
            creds = service_account.Credentials.from_service_account_info(sa_info)
            _gcs_client = storage.Client(credentials=creds)
        else:
            _gcs_client = storage.Client()
        _gcs_bucket = _gcs_client.bucket(gcs_bucket_name)
        return True
    except Exception as e:
        st.error(f"Failed to initialize GCS: {e}")
        return False


def _gcs_path(job_id: str, *parts: str) -> str:
    prefix = gcs_prefix.strip("/")
    return "/".join([p.strip("/") for p in [prefix, job_id, *parts] if p])


def gcs_upload_json(job_id: str, relpath: str, obj: dict):
    blob = _gcs_bucket.blob(_gcs_path(job_id, relpath))
    blob.upload_from_string(json.dumps(obj), content_type="application/json")


def gcs_download_json(job_id: str, relpath: str):
    try:
        blob = _gcs_bucket.blob(_gcs_path(job_id, relpath))
        if not blob.exists(_gcs_client):
            return None
        return json.loads(blob.download_as_text())
    except Exception:
        return None


def gcs_list(job_id: str, relprefix: str):
    prefix = _gcs_path(job_id, relprefix).rstrip("/") + "/"
    return list(_gcs_client.list_blobs(gcs_bucket_name, prefix=prefix))


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
    (?:in\s+)?                 # optional 'in '
    ((?:19|20)\d{2})           # base 4-digit year (capture)
    (?:\s*/\s*\d{2}          # ' /25' short range
       |-(?:19|20)\d{2}        # or '-2025' full range
    )?                          # optional year-range
    \)?\s*$                    # optional ')' then end of string
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
    data = getattr(resp, "data", None) or resp["data"]
    out = []
    for d in data:
        emb = getattr(d, "embedding", None)
        if emb is None:
            emb = d.get("embedding", [])
        out.append(np.array(emb, dtype=float))
    return out

def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def dedupe_embeddings(queries, threshold=0.86, model="text-embedding-3-small"):
    if not queries:
        return [], {}
    try:
        vecs = get_embeddings(queries, model=model)
    except Exception as e:
        st.warning(f"Embedding dedupe skipped (embedding error): {e}")
        return queries, {q: [q] for q in queries}
    kept, groups, removed = [], {}, set()
    for i, qi in enumerate(queries):
        if i in removed: continue
        kept.append(qi)
        groups[qi] = [qi]
        for j in range(i+1, len(queries)):
            if j in removed: continue
            sim = cosine(vecs[i], vecs[j])
            if sim >= threshold:
                groups[qi].append(queries[j])
                removed.add(j)
    return kept, groups

def dedupe_pipeline(
    queries,
    use_exact=True,
    fuzzy_ratio=92,
    use_embed=True,
    embed_threshold=0.86,
    embed_model="text-embedding-3-small"
):
    if not queries:
        return [], {}
    q = queries[:]
    if use_exact:
        q, groups_exact = dedupe_exact(q)
    else:
        groups_exact = {x: [x] for x in q}
    q2, groups_ts = dedupe_token_set(q, min_ratio=fuzzy_ratio)
    merged_ts = {rep: [] for rep in q2}
    for rep in q2:
        merged_ts[rep].extend(groups_ts[rep])
    if use_embed and len(q2) > 1:
        reps, sem_groups = dedupe_embeddings(q2, threshold=embed_threshold, model=embed_model)
        final_groups = {r: [] for r in reps}
        for r in reps:
            for member in sem_groups[r]:
                final_groups[r].extend(merged_ts.get(member, [member]))
        for r in final_groups:
            seen, ded = set(), []
            for item in final_groups[r]:
                if item not in seen:
                    seen.add(item); ded.append(item)
            final_groups[r] = ded
        return reps, final_groups
    return q2, merged_ts

# =========================
# FETCH FAN-OUTS (parametrized + retries)
# =========================

def fetch_query_fan_outs_multi(text, attempts=1, temp=0.0, cand_count=None, timeout_s=60):
    """Generate fan-out queries via Gemini, with retries on ReadTimeout."""
    cand = cand_count if cand_count is not None else candidate_count
    queries = []
    for _ in range(attempts):
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": temp, "candidateCount": cand},
        }
        response = None
        for retry in range(2):
            try:
                response = requests.post(endpoint, json=payload, timeout=timeout_s)
                response.raise_for_status()
                break
            except ReadTimeout:
                time.sleep(0.8)
            except Exception as e:
                st.warning(f"Fan-out fetch failed for '{text}': {e}")
                break
        if not response:
            continue
        try:
            data = response.json().get("candidates", [])
            for cand in data:
                queries.extend(cand.get("groundingMetadata", {}).get("webSearchQueries", []) or [])
        except Exception as e:
            st.warning(f"Error parsing fan-out response JSON for '{text}': {e}")
    return queries

# Parallel Level-2 expansion

def expand_level2_parallel(level1, attempts=1, temp=0.4, max_workers=6, cand_count=4, timeout_s=15):
    results = []
    if not level1:
        return results
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(level1)))) as ex:
        futs = {
            ex.submit(fetch_query_fan_outs_multi, q, attempts, temp, cand_count, timeout_s): q
            for q in level1
        }
        for fut in as_completed(futs):
            try:
                results.extend(fut.result() or [])
            except Exception as e:
                st.warning(f"Level-2 expansion error: {e}")
    return results

# =========================
# CONTENT EXTRACTION (fast path first) + caching
# =========================

def extract_content_fast(url):
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    r = scraper.get(url, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def extract_content_playwright(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        resp = page.goto(url, timeout=45000)
        if resp and resp.status == 403:
            raise RuntimeError("HTTP 403 Forbidden")
        html = page.content()
        browser.close()
    return BeautifulSoup(html, "html.parser")

@st.cache_data(show_spinner=False, ttl=3600)
def cached_extract(url):
    try:
        soup = extract_content_fast(url)
        if not soup.find("h1"):
            soup = extract_content_playwright(url)
    except Exception:
        soup = extract_content_playwright(url)
    h1_tag = soup.find("h1")
    h1_text = h1_tag.get_text(strip=True) if h1_tag else ""
    headings = [(tag.name.upper(), tag.get_text(strip=True)) for tag in soup.find_all(["h2","h3","h4"])]
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    list_items = [li.get_text(strip=True) for li in soup.find_all("li")]
    body = "\n".join(paragraphs + list_items)
    return h1_text, headings, body, None

@st.cache_data(show_spinner=False, ttl=86400)
def cached_fanouts(text, cand_count, temp, timeout_s):
    return fetch_query_fan_outs_multi(text, attempts=attempts, temp=temp, cand_count=cand_count, timeout_s=timeout_s)

# =========================
# PROMPT BUILDING + GPT CALL (batched)
# =========================

def build_prompt(h1, headings, body, queries):
    lines = [
        "I’m auditing this page for content gaps.",
        f"Main topic (H1): {h1}",
        "",
        "Existing Headings:",
    ]
    for lvl, txt in headings:
        lines.append(f"- {lvl}: {txt}")
    lines += ["", "Page Body Text:", body[:BODY_CHAR_LIMIT], "", "Queries to check coverage:"]
    for q in queries:
        lines.append(f"- {q}")
    lines += [
        "",
        "Please provide coverage entries for *all* of the above queries, even if covered=false.",
        "",
        "Given the above headings and body text, return ONLY a JSON array with keys:",
        "query (string), covered (true/false), explanation (string).",
        "Example: [{\"query\":\"...\",\"covered\":true,\"explanation\":\"...\"}]"
    ]
    return "\n".join(lines)


def get_explanations(prompt, temperature=0.1, max_retries=2):
    system_msg = "You are an SEO content gap auditor. Output ONLY a JSON array."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    last_resp = ""
    for _ in range(max_retries):
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=temperature
            )
            text = resp.choices[0].message.content.strip()
            last_resp = text
            match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            time.sleep(0.8)
    st.warning(f"Failed to parse OpenAI response as JSON. Raw response: {last_resp}")
    return []


def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def get_explanations_batched(h1, headings, body, queries, batch_size=8, temperature=0.1):
    out = []
    for group in chunked(queries, batch_size):
        prompt = build_prompt(h1, headings, body, group)
        out.extend(get_explanations(prompt, temperature=temperature))
    return out

# =========================
# JOB LIFECYCLE HELPERS (GCS-backed)
# =========================

def create_or_load_job(urls, settings, resume=False, job_id=None):
    """Create a new job or load an existing one from GCS. Returns (job_id, config)."""
    if not use_gcs:
        # In-memory only
        job_id = job_id or uuid.uuid4().hex[:12]
        return job_id, {"urls": urls, "settings": settings}

    if not _init_gcs():
        job_id = job_id or uuid.uuid4().hex[:12]
        return job_id, {"urls": urls, "settings": settings}

    if resume:
        if not job_id:
            st.error("Provide a Job ID to resume.")
            st.stop()
        cfg = gcs_download_json(job_id, "config.json")
        if not cfg:
            st.error("No config found for that Job ID. Check bucket/prefix.")
            st.stop()
        return job_id, cfg
    else:
        job_id = job_id or uuid.uuid4().hex[:12]
        cfg = {"urls": urls, "settings": settings}
        gcs_upload_json(job_id, "config.json", cfg)
        gcs_upload_json(job_id, "progress.json", {"done": 0, "total": len(urls), "status": "started"})
        return job_id, cfg


def load_existing_results_to_state(job_id):
    """Prefill session_state tables from prior GCS outputs so the UI shows previous progress."""
    if not use_gcs or not _gcs_bucket:
        return set()
    # Load detailed
    processed_idx = set()
    for blob in gcs_list(job_id, "detailed"):
        try:
            data = json.loads(blob.download_as_text())
            st.session_state.detailed.append(data)
            # file name pattern detailed/00012.json
            bname = blob.name.split("/")[-1]
            idx = int(bname.split(".")[0])
            processed_idx.add(idx)
        except Exception:
            pass
    # Load summary
    for blob in gcs_list(job_id, "summary"):
        try:
            data = json.loads(blob.download_as_text())
            st.session_state.summary.append(data)
        except Exception:
            pass
    # Load actions
    for blob in gcs_list(job_id, "actions"):
        try:
            data = json.loads(blob.download_as_text())
            st.session_state.actions.append(data)
        except Exception:
            pass
    # Load skipped
    for blob in gcs_list(job_id, "skipped"):
        try:
            data = json.loads(blob.download_as_text())
            st.session_state.skipped.append(data)
        except Exception:
            pass
    return processed_idx


def persist_step(job_id, idx, detailed_row=None, summary_row=None, actions_row=None, skipped_row=None, progress=None):
    if not use_gcs or not _gcs_bucket:
        return
    if detailed_row is not None:
        gcs_upload_json(job_id, f"detailed/{idx:05d}.json", detailed_row)
    if summary_row is not None:
        gcs_upload_json(job_id, f"summary/{idx:05d}.json", summary_row)
    if actions_row is not None:
        gcs_upload_json(job_id, f"actions/{idx:05d}.json", actions_row)
    if skipped_row is not None:
        gcs_upload_json(job_id, f"skipped/{idx:05d}.json", skipped_row)
    if progress is not None:
        gcs_upload_json(job_id, "progress.json", progress)

# =========================
# MAIN AUDIT LOOP
# =========================

# Prepare URLs (from UI); total computed after job config (could be overridden on resume)
urls_ui = [u.strip() for u in urls_input.splitlines() if u.strip()]

start_clicked = st.button("Start Audit (resume-capable)")

if start_clicked and (urls_ui or resume_existing) and not st.session_state.processed:
    # Collect settings snapshot to persist with job
    settings = dict(
        enable_dedupe=enable_dedupe,
        fuzzy_ratio=fuzzy_ratio,
        embed_on=embed_on,
        embed_threshold=embed_threshold,
        embed_model=embed_model,
        max_workers=max_workers,
        lvl2_candidates=lvl2_candidates,
        lvl2_timeout=lvl2_timeout,
        lvl1_timeout=lvl1_timeout,
        gpt_batch_size=gpt_batch_size,
        normalize_year_suffix=normalize_year_suffix,
        gemini_temp=gemini_temp,
        gpt_temp=gpt_temp,
        attempts=attempts,
        candidate_count=candidate_count,
    )

    # Create or load job
    job_id = job_id_input.strip() or None
    job_id, cfg = create_or_load_job(urls_ui, settings, resume=resume_existing, job_id=job_id)
    st.session_state.active_job_id = job_id
    st.info(f"Job ID: {job_id}")

    # Use URLs from saved config when resuming
    urls = cfg.get("urls", urls_ui)
    total = len(urls)

    # Preload previous outputs
    processed_idx = set()
    if use_gcs and _init_gcs():
        processed_idx = load_existing_results_to_state(job_id)

    progress_bar = st.progress(0)
    status_text  = st.empty()
    start_time   = time.time()

    # Prepare fresh accumulators for this run
    # (already prefilled above if resuming)

    done_so_far = len(processed_idx)

    for idx, url in enumerate(urls):
        # Skip already processed indices
        if idx in processed_idx:
            continue

        elapsed = time.time() - start_time
        # avoid div by zero if none processed this run yet
        denom = max(1, (idx - min(processed_idx) if processed_idx else idx + 1))
        eta = int((elapsed/denom) * (total - idx))
        status_text.text(f"Processing {idx+1}/{total} — ETA: {eta}s")
        progress_bar.progress(int(((idx+1) / total) * 100))

        # Content extraction (cached, fast-path first)
        try:
            h1_text, headings, body, err = cached_extract(url)
        except Exception as e:
            reason = f"Fetch error ({e})"
            st.warning(f"Skipped {url}: {reason}")
            skip_row = {"Address": url, "Reason": "Fetch error"}
            st.session_state.skipped.append(skip_row)
            persist_step(job_id, idx, skipped_row=skip_row, progress={"done": done_so_far, "total": total, "status": "running", "last": url})
            continue

        if err or not h1_text:
            reason = err or "No H1 found"
            st.warning(f"Skipped {url}: {reason}")
            skip_row = {"Address": url, "Reason": reason}
            st.session_state.skipped.append(skip_row)
            persist_step(job_id, idx, skipped_row=skip_row, progress={"done": done_so_far, "total": total, "status": "running", "last": url})
            continue

        # Level 1 fan-out (cached)
        lvl1 = st.session_state.h1_fanout_cache.get(h1_text) or cached_fanouts(h1_text, 7, gemini_temp, lvl1_timeout)
        st.session_state.h1_fanout_cache[h1_text] = lvl1

        # Level 2 fan-out (parallel & lighter)
        level2_key = ("__lvl2__", h1_text, lvl2_candidates, lvl2_timeout)
        level2 = st.session_state.fanout_layer2_cache.get(level2_key)
        if level2 is None:
            level2 = expand_level2_parallel(
                lvl1,
                attempts=attempts,
                temp=gemini_temp,
                max_workers=max_workers,
                cand_count=lvl2_candidates,
                timeout_s=lvl2_timeout
            )
            st.session_state.fanout_layer2_cache[level2_key] = level2

        all_qs = lvl1 + level2

        # Normalize queries by stripping trailing years (optional, before dedupe)
        if normalize_year_suffix:
            all_qs = [strip_trailing_years(q) for q in all_qs]
            all_qs = [q for q in all_qs if q]

        if not all_qs:
            reason = "No queries generated"
            st.warning(f"Skipped {url}: {reason}")
            skip_row = {"Address": url, "Reason": reason}
            st.session_state.skipped.append(skip_row)
            persist_step(job_id, idx, skipped_row=skip_row, progress={"done": done_so_far, "total": total, "status": "running", "last": url})
            continue

        # --- Dedupe (optional) ---
        raw_queries = all_qs[:]  # keep a copy for transparency
        if enable_dedupe:
            reps, groups = dedupe_pipeline(
                all_qs,
                use_exact=True,
                fuzzy_ratio=fuzzy_ratio,
                use_embed=embed_on,
                embed_threshold=embed_threshold,
                embed_model=embed_model
            )
            queries_for_prompt = reps
            grouped_view = {rep: "; ".join(members) for rep, members in groups.items()}
        else:
            queries_for_prompt = raw_queries
            grouped_view = {q: q for q in raw_queries}

        # Build prompt & call GPT (batched)
        results = get_explanations_batched(
            h1_text, headings, body, queries_for_prompt, batch_size=gpt_batch_size, temperature=gpt_temp
        )
        if not results:
            reason = "No usable output from OpenAI"
            st.warning(f"Skipped {url}: {reason}")
            skip_row = {"Address": url, "Reason": reason}
            st.session_state.skipped.append(skip_row)
            persist_step(job_id, idx, skipped_row=skip_row, progress={"done": done_so_far, "total": total, "status": "running", "last": url})
            continue

        # Summaries & Actions
        covered = sum(1 for r in results if r.get("covered"))
        pct = int((covered / len(results)) * 100) if results else 0
        summary_row = {
            "Address": url,
            "Fan-out Count (raw)": len(raw_queries),
            "Queries Used (after dedupe)": len(queries_for_prompt),
            "Coverage (%)": pct
        }
        st.session_state.summary.append(summary_row)
        missing = [r.get("query") for r in results if not r.get("covered")]
        actions_row = {
            "Address": url,
            "Recommended Sections to Add": "; ".join(missing)
        }
        st.session_state.actions.append(actions_row)

        # Detailed row
        row = {
            "Address": url,
            "H1": h1_text,
            "Headings": " | ".join(f"{l}:{t}" for l, t in headings),
            "All Queries (raw)": "; ".join(raw_queries),
            "Queries Used (final)": "; ".join(queries_for_prompt),
        }
        if enable_dedupe:
            row["Dedupe Groups"] = " || ".join([f"{rep} => {members}" for rep, members in grouped_view.items()])
        for i, r in enumerate(results, start=1):
            row[f"Query {i}"]       = r.get("query")
            row[f"Covered {i}"]     = r.get("covered")
            row[f"Explanation {i}"] = r.get("explanation")
        st.session_state.detailed.append(row)

        # Persist this step to GCS
        done_so_far += 1
        persist_step(job_id, idx, detailed_row=row, summary_row=summary_row, actions_row=actions_row,
                     progress={"done": done_so_far, "total": total, "status": "running", "last": url})

    # Finalize
    if use_gcs and _gcs_bucket:
        persist_step(job_id, total, progress={"done": total, "total": total, "status": "done"})

    st.session_state.processed = True

# --- Display / Download Results ---
if st.session_state.processed:
    st.header("Results")
    if st.session_state.detailed:
        st.subheader("Detailed")
        df = pd.DataFrame(st.session_state.detailed)
        st.download_button("Download Detailed CSV", df.to_csv(index=False).encode("utf-8"), "detailed.csv", "text/csv")
        st.dataframe(df)
    if st.session_state.summary:
        st.subheader("Summary")
        df = pd.DataFrame(st.session_state.summary)
        st.download_button("Download Summary CSV", df.to_csv(index=False).encode("utf-8"), "summary.csv", "text/csv")
        st.dataframe(df)
    if st.session_state.actions:
        st.subheader("Actions")
        df = pd.DataFrame(st.session_state.actions)
        st.download_button("Download Actions CSV", df.to_csv(index=False).encode("utf-8"), "actions.csv", "text/csv")
        st.dataframe(df)
    if st.session_state.skipped:
        st.subheader("Skipped URLs & Reasons")
        st.table(pd.DataFrame(st.session_state.skipped))



















