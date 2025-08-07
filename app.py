import subprocess
import sys
# Ensure Playwright is installed for browser automation
tools_install = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)

import time
import streamlit as st
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import cloudscraper
import openai

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Overview/AI Mode query fan-out gap analysis", layout="wide")
st.title("🔍 AI Overview/AI Mode Query Fan-Out Gap Analysis")

st.sidebar.header("About this tool")
st.sidebar.write(
    """
This tool identifies content gaps by:
1. Extracting your page's H1 headings.
2. Generating user query fan-outs for each H1 query using Google Gemini.
3. Performing a second-level fan-out on those queries to maximize coverage.
4. Comparing all generated queries against the page's headings to highlight missing topics.
"""
)

# --- Fixed Configuration ---
gemini_temp = 0.4      # Diversity for fan-out generation
gpt_temp = 0.1         # Temperature for gap reasoning
attempts = 1           # Number of Gemini aggregation calls
candidate_count = 7    # Number of candidates per Gemini call

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

# --- Helper: Fetch Fan-Out Queries via Gemini ---
def fetch_query_fan_outs_multi(text, attempts=1, temp=0.0):
    queries = []
    for _ in range(attempts):
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": temp, "candidateCount": candidate_count},
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            candidates = response.json().get("candidates", [])
            for cand in candidates:
                queries.extend(
                    cand.get("groundingMetadata", {}).get("webSearchQueries", []) or []
                )
        except Exception as e:
            st.warning(f"Fan-out fetch failed for '{text}': {e}")
    return queries

# --- Helper: Extract H1 and Subheadings ---
def extract_h1_and_headings(url):
    # Try Playwright first for JS-rendered pages
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            resp = page.goto(url, timeout=60000)
            if resp and resp.status == 403:
                return "", [], "HTTP 403 Forbidden"
            html = page.content()
            browser.close()
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        # Fallback to cloudscraper if Playwright fails
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            r = scraper.get(url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception:
            return "", [], "Fetch error"
    h1 = soup.find("h1")
    h1_text = h1.get_text(strip=True) if h1 else ""
    headings = [
        (tag.name.upper(), tag.get_text(strip=True))
        for tag in soup.find_all(["h2", "h3", "h4"])
    ]
    return h1_text, headings, None

# --- Helper: Build the OpenAI Prompt ---
def build_prompt(h1, headings, queries):
    lines = [
        "I’m auditing this page for content gaps.",
        f"Main topic (H1): {h1}",
        "",
        "1) Existing headings (in order):",
    ]
    for lvl, txt in headings:
        lines.append(f"{lvl}: {txt}")
    lines.append("")
    lines.append("2) User queries to cover:")
    for q in queries:
        lines.append(f"- {q}")
    lines.append("")
    lines.append(
        "3) Return a JSON array with keys: query (string), covered (true/false), explanation (string)."
    )
    lines.append(
        'Example: [{"query":"...","covered":true,"explanation":"..."}]'
    )
    return "\n".join(lines)

# --- Helper: Call OpenAI for Coverage Analysis ---
def get_explanations(prompt, temperature=0.1, max_retries=2):
    system_msg = (
        "You are an SEO content gap auditor. Output ONLY a JSON array. "
        "Each item must have exactly: query, covered, explanation."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(max_retries):
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=temperature
            )
            content = resp.choices[0].message.content.strip().strip("```")
            return json.loads(content)
        except Exception:
            time.sleep(1)
    st.warning("Failed to parse OpenAI response as JSON.")
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
        avg_per = elapsed / idx
        remaining = total - idx
        eta = int(remaining * avg_per)
        status_text.text(f"Processing {idx}/{total} — ETA: {eta}s")
        progress_bar.progress(int(idx/total * 100))

        # Extract headings
        h1_text, headings, err = extract_h1_and_headings(url)
        if err or not h1_text:
            reason = err or "No H1 found"
            st.warning(f"Skipped {url}: {reason}")
            st.session_state.skipped.append({"Address": url, "Reason": reason})
            continue

        # --- Level 1: H1 Fan-Out ---
        if h1_text in st.session_state.h1_fanout_cache:
            level1_queries = st.session_state.h1_fanout_cache[h1_text]
        else:
            level1_queries = fetch_query_fan_outs_multi(
                h1_text, attempts=attempts, temp=gemini_temp
            )
            st.session_state.h1_fanout_cache[h1_text] = level1_queries

        # --- Level 2: Second-Pass Fan-Out ---
        all_queries = []
        for q in level1_queries:
            if q in st.session_state.fanout_layer2_cache:
                second_queries = st.session_state.fanout_layer2_cache[q]
            else:
                second_queries = fetch_query_fan_outs_multi(
                    q, attempts=attempts, temp=gemini_temp
                )
                st.session_state.fanout_layer2_cache[q] = second_queries
            all_queries.extend(second_queries)
        all_queries = level1_queries + all_queries

        if not all_queries:
            st.warning(f"Skipped {url}: no fan-out queries generated.")
            st.session_state.skipped.append({"Address": url, "Reason": "No queries generated"})
            continue

        # --- Gap Analysis via OpenAI ---
        prompt = build_prompt(h1_text, headings, all_queries)
        results = get_explanations(prompt, temperature=gpt_temp)
        if not results:
            st.warning(f"Skipped {url}: OpenAI returned no usable output.")
            st.session_state.skipped.append({"Address": url, "Reason": "No results from OpenAI"})
            continue

        # --- Summaries & Actions ---
        covered_count = sum(1 for r in results if r.get("covered"))
        coverage_pct = int((covered_count / len(results)) * 100)
        st.session_state.summary.append({
            "Address": url,
            "Fan-out Count": len(all_queries),
            "Coverage (%)": coverage_pct
        })
        missing_sections = [r.get("query") for r in results if not r.get("covered")]
        st.session_state.actions.append({
            "Address": url,
            "Recommended Sections to Add": "; ".join(missing_sections)
        })

        # --- Detailed Row Construction ---
        row = {
            "Address": url,
            "H1": h1_text,
            "Headings": " | ".join(f"{lvl}:{txt}" for lvl, txt in headings)
        }
        for i, r in enumerate(results, start=1):
            row[f"Query {i}"] = r.get("query")
            row[f"Covered {i}"] = r.get("covered")
            row[f"Explanation {i}"] = r.get("explanation")
        st.session_state.detailed.append(row)

    progress_bar.progress(100)
    status_text.text("Audit Complete!")
    st.session_state.processed = True

# --- Display / Download Results ---
if st.session_state.processed:
    st.header("Results")
    # Detailed Table
    if st.session_state.detailed:
        st.subheader("Detailed")
        df_detailed = pd.DataFrame(st.session_state.detailed)
        st.download_button(
            "Download Detailed CSV",
            df_detailed.to_csv(index=False).encode("utf-8"),
            "detailed.csv", "text/csv"
        )
        st.dataframe(df_detailed)
    # Summary Table
    if st.session_state.summary:
        st.subheader("Summary")
        df_summary = pd.DataFrame(st.session_state.summary)
        st.download_button(
            "Download Summary CSV",
            df_summary.to_csv(index=False).encode("utf-8"),
            "summary.csv", "text/csv"
        )
        st.dataframe(df_summary)
    # Actions Table
    if st.session_state.actions:
        st.subheader("Actions")
        df_actions = pd.DataFrame(st.session_state.actions)
        st.download_button(
            "Download Actions CSV",
            df_actions.to_csv(index=False).encode("utf-8"),
            "actions.csv", "text/csv"
        )
        st.dataframe(df_actions)
    # Skipped URLs
    if st.session_state.skipped:
        st.subheader("Skipped URLs & Reasons")
        df_skipped = pd.DataFrame(st.session_state.skipped)
        st.table(df_skipped)










