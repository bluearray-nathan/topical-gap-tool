import subprocess
import sys
# Ensure Playwright is installed for browser automation
tools_install = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)

import time
import streamlit as st
import pandas as pd
import requests
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import cloudscraper
import openai
from requests.exceptions import ReadTimeout

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Overview/AI Mode query fan-out gap analysis", layout="wide")
st.title("🔍 AI Overview/AI Mode Query Fan-Out Gap Analysis")

st.sidebar.header("About this tool")
st.sidebar.write(
    """
This tool identifies content gaps by:
1. Extracting your page's H1 headings and body text.
2. Generating user query fan-outs for each H1 query using Google Gemini.
3. Performing a second-level fan-out on those queries to maximize coverage.
4. Comparing all generated queries against the page's full content to highlight missing topics.
"""
)

# --- Fixed Configuration ---
gemini_temp = 0.4      # Diversity for fan-out generation
gpt_temp    = 0.1      # Temperature for gap reasoning
attempts    = 1        # Number of Gemini aggregation calls
candidate_count = 7    # Number of candidates per call
BODY_CHAR_LIMIT = 2000 # Limit body text passed to GPT

# --- Normalization (Sidebar) ---
st.sidebar.subheader("Normalization")
normalize_year_suffix = st.sidebar.checkbox("Strip trailing years (e.g., '2024' or '2024/25')", value=True)

# --- Load API Keys from Streamlit Secrets ---
openai.api_key = st.secrets["openai"]["api_key"]
gemini_api_key = st.secrets["google"]["gemini_api_key"]

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
total = len(urls)
if total > 0:
    st.write(f"Found {total} URLs to process.")

# --- Reset State When URLs Change ---
if urls and st.session_state.last_urls != urls:
    st.session_state.last_urls = urls
    st.session_state.processed = False
    st.session_state.detailed.clear()
    st.session_state.summary.clear()
    st.session_state.actions.clear()
    st.session_state.skipped.clear()
    st.session_state.h1_fanout_cache.clear()
    st.session_state.fanout_layer2_cache.clear()

# --- Helper: Strip trailing year(s) from queries ---
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

# --- Helper: Fetch Fan-Out Queries via Gemini ---
def fetch_query_fan_outs_multi(text, attempts=1, temp=0.0):
    """
    Generate fan-out queries via Gemini, with retries on ReadTimeout.
    """
    queries = []
    for attempt_i in range(attempts):
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": temp, "candidateCount": candidate_count},
        }
        # Retry loop for transient read timeouts
        response = None
        for retry in range(3):
            try:
                response = requests.post(endpoint, json=payload, timeout=60)
                response.raise_for_status()
                break
            except ReadTimeout:
                st.warning(f"ReadTimeout on fan-out fetch for '{text}' (attempt {retry+1}/3), retrying...")
                time.sleep(1)
                continue
            except Exception as e:
                st.warning(f"Fan-out fetch failed for '{text}': {e}")
                break
        if not response:
            continue
        try:
            data = response.json().get("candidates", [])
            for cand in data:
                queries.extend(
                    cand.get("groundingMetadata", {}).get("webSearchQueries", []) or []
                )
        except Exception as e:
            st.warning(f"Error parsing fan-out response JSON for '{text}': {e}")
    return queries

# --- Helper: Extract H1, Headings, and Body Text ---
def extract_content(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            resp = page.goto(url, timeout=60000)
            if resp and resp.status == 403:
                return "", [], "", "HTTP 403 Forbidden"
            html = page.content()
            browser.close()
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser":"chrome","platform":"windows","mobile":False}
            )
            r = scraper.get(url, timeout=60)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception:
            return "", [], "", "Fetch error"
    # H1 and subheadings
    h1_tag = soup.find("h1")
    h1_text = h1_tag.get_text(strip=True) if h1_tag else ""
    headings = [(tag.name.upper(), tag.get_text(strip=True)) for tag in soup.find_all(["h2","h3","h4"])]
    # Body text: paragraphs and list items
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    list_items = [li.get_text(strip=True) for li in soup.find_all("li")]
    body = "\n".join(paragraphs + list_items)
    return h1_text, headings, body, None

# --- Helper: Build the OpenAI Prompt with Full Content ---
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

# --- Helper: Call OpenAI for Coverage Analysis ---
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
            time.sleep(1)
    st.warning(f"Failed to parse OpenAI response as JSON. Raw response: {last_resp}")
    return []

# --- Main Audit Loop ---
if st.button("Start Audit") and urls and not st.session_state.processed:
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    # Clear previous results
    st.session_state.detailed.clear()
    st.session_state.summary.clear()
    st.session_state.actions.clear()
    st.session_state.skipped.clear()

    for idx, url in enumerate(urls, start=1):
        elapsed = time.time() - start_time
        eta = int((elapsed/idx) * (total-idx))
        status_text.text(f"Processing {idx}/{total} — ETA: {eta}s")
        progress_bar.progress(int(idx/total*100))

        h1_text, headings, body, err = extract_content(url)
        if err or not h1_text:
            reason = err or "No H1 found"
            st.warning(f"Skipped {url}: {reason}")
            st.session_state.skipped.append({"Address": url, "Reason": reason})
            continue

        # Level 1 fan-out
        lvl1 = st.session_state.h1_fanout_cache.get(h1_text) or fetch_query_fan_outs_multi(
            h1_text, attempts, gemini_temp
        )
        st.session_state.h1_fanout_cache[h1_text] = lvl1

        # Level 2 fan-out
        all_qs = []
        for q in lvl1:
            sub = st.session_state.fanout_layer2_cache.get(q) or fetch_query_fan_outs_multi(
                q, attempts, gemini_temp
            )
            st.session_state.fanout_layer2_cache[q] = sub
            all_qs.extend(sub)
        all_qs = lvl1 + all_qs

        # Normalize queries by stripping trailing years (optional)
        if normalize_year_suffix:
            all_qs = [strip_trailing_years(q) for q in all_qs]
            # drop any accidental empties after stripping
            all_qs = [q for q in all_qs if q]

        if not all_qs:
            st.warning(f"Skipped {url}: no queries generated.")
            st.session_state.skipped.append({"Address": url, "Reason": "No queries generated"})
            continue

        prompt = build_prompt(h1_text, headings, body, all_qs)
        results = get_explanations(prompt, temperature=gpt_temp)
        if not results:
            st.warning(f"Skipped {url}: OpenAI returned no usable output.")
            st.session_state.skipped.append({"Address": url, "Reason": "No usable output from OpenAI"})
            continue

        covered = sum(1 for r in results if r.get("covered"))
        pct = int((covered / len(results)) * 100)
        st.session_state.summary.append({"Address": url, "Fan-out Count": len(all_qs), "Coverage (%)": pct})
        missing = [r.get("query") for r in results if not r.get("covered")]
        st.session_state.actions.append({"Address": url, "Recommended Sections to Add": "; ".join(missing)})

        row = {
            "Address": url,
            "H1": h1_text,
            "Headings": " | ".join(f"{l}:{t}" for l, t in headings),
            "All Queries": "; ".join(all_qs)  # (normalized list used for GPT)
        }
        for i, r in enumerate(results, start=1):
            row[f"Query {i}"]       = r.get("query")
            row[f"Covered {i}"]     = r.get("covered")
            row[f"Explanation {i}"] = r.get("explanation")
        st.session_state.detailed.append(row)

    progress_bar.progress(100)
    status_text.text("Audit Complete!")
    st.session_state.processed = True

# --- Display / Download Results ---
if st.session_state.processed:
    st.header("Results")
    if st.session_state.detailed:
        st.subheader("Detailed")
        df = pd.DataFrame(st.session_state.detailed)
        st.download_button("Download Detailed CSV", df.to_csv(index=False).encode("utf-8"), "detailed.csv")
        st.dataframe(df)
    if st.session_state.summary:
        st.subheader("Summary")
        df = pd.DataFrame(st.session_state.summary)
        st.download_button("Download Summary CSV", df.to_csv(index=False).encode("utf-8"), "summary.csv")
        st.dataframe(df)
    if st.session_state.actions:
        st.subheader("Actions")
        df = pd.DataFrame(st.session_state.actions)
        st.download_button("Download Actions CSV", df.to_csv(index=False).encode("utf-8"), "actions.csv")
        st.dataframe(df)
    if st.session_state.skipped:
        st.subheader("Skipped URLs & Reasons")
        st.table(pd.DataFrame(st.session_state.skipped))

















