import subprocess
import sys
# Install Playwright browser binaries if possible (non-fatal)
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)

import time
import streamlit as st
import pandas as pd
import requests
import re
import json
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import cloudscraper
import openai

# --- Streamlit UI ---
st.set_page_config(page_title="Content Gap Audit", layout="wide")
st.title("🔍 Content Gap Audit Tool")

# Sidebar explanation
st.sidebar.header("About Content Gap Audit")
st.sidebar.write(
    """This tool automates an SEO content coverage audit by:
1. Extracting your page's H1 and subheadings (H2–H4).
2. Using Google Gemini to generate relevant user queries (fan-outs).
3. Comparing queries against your headings with OpenAI GPT to identify missing topics."""
)

# Load API keys
openai.api_key = st.secrets["openai"]["api_key"]
gemini_api_key = st.secrets["google"]["gemini_api_key"]

# Input URLs directly
urls_input = st.text_area(
    "Enter one URL per line to analyze headers and content gaps:",
    placeholder="https://example.com/page1\nhttps://example.com/page2"
)

# Red button styling
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #e63946;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state if needed
if "last_urls" not in st.session_state:
    st.session_state.last_urls = []
    st.session_state.processed = False
    st.session_state.detailed = []
    st.session_state.summary = []
    st.session_state.actions = []
    st.session_state.skipped = []

# Parse URLs
urls = [u.strip() for u in urls_input.splitlines() if u.strip()] if urls_input else []
total = len(urls)
if urls:
    st.write(f"Found {total} URLs to process.")

# If the input changed, reset processed state
if urls and st.session_state.last_urls != urls:
    st.session_state.last_urls = urls
    st.session_state.processed = False
    st.session_state.detailed = []
    st.session_state.summary = []
    st.session_state.actions = []
    st.session_state.skipped = []

# Trigger
start_clicked = st.button("Start Audit", key="start")

# Only run audit if requested and not already done for this input
if start_clicked and urls:
    st.session_state.processed = False  # force fresh run

if urls and not st.session_state.processed:
    # Begin processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    # Clear previous just in case
    st.session_state.detailed = []
    st.session_state.summary = []
    st.session_state.actions = []
    st.session_state.skipped = []

    # Helper: extract H1 + headings with fallback
    def extract_h1_and_headings(url):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                resp = page.goto(url, timeout=60000)
                if resp and resp.status == 403:
                    return "", [], "HTTP 403 Forbidden (Playwright)"
                html = page.content()
                browser.close()
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            try:
                scraper = cloudscraper.create_scraper(
                    browser={"browser": "chrome", "platform": "windows", "mobile": False}
                )
                r = scraper.get(url, timeout=30)
                try:
                    r.raise_for_status()
                except requests.exceptions.HTTPError as he:
                    code = he.response.status_code if he.response else "unknown"
                    reason = he.response.reason if he.response and hasattr(he.response, "reason") else ""
                    return "", [], f"HTTP {code} {reason} (fallback)"
                soup = BeautifulSoup(r.text, "html.parser")
            except Exception:
                return "", [], "Fetch failed (both Playwright and fallback)"
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        headings = [(tag.name.upper(), tag.get_text(strip=True)) for tag in soup.find_all(["h2", "h3", "h4"])]
        return h1, headings, None

    # Helper: fetch fan-out queries with low randomness
    def fetch_query_fan_outs(h1_text):
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": h1_text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.0},  # deterministic fan-outs
        }
        try:
            r = requests.post(endpoint, json=payload, timeout=30)
            r.raise_for_status()
            cand = r.json().get("candidates", [{}])[0]
            return cand.get("groundingMetadata", {}).get("webSearchQueries", [])
        except Exception as e:
            st.warning(f"Fan-out fetch failed: {e}")
            return []

    # Helper: build prompt
    def build_prompt(h1, headings, queries):
        lines = [
            "I’m auditing this page for content gaps.",
            f"Main topic (H1): {h1}",
            "",
            "1) Existing headings (in order):",
        ]
        for lvl, txt in headings:
            lines.append(f"{lvl}: {txt}")
        lines.extend(["", "2) User queries to cover:"])
        for q in queries:
            lines.append(f"- {q}")
        lines.extend(
            [
                "",
                "3) Return JSON array of objects with keys: query, covered, explanation.",
                'Example: [{"query":"...","covered":true,"explanation":"..."}]',
            ]
        )
        return "\n".join(lines)

    # Helper: call OpenAI
    def get_explanations(prompt):
        resp = openai.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        txt = resp.choices[0].message.content.strip()
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
        try:
            arr = json.loads(txt)
            return arr if isinstance(arr, list) else []
        except:
            return []

    # Loop through URLs
    for idx, url in enumerate(urls):
        elapsed = time.time() - start_time
        avg = elapsed / (idx + 1)
        remaining = total - (idx + 1)
        eta_secs = remaining * avg
        mins = int(eta_secs // 60)
        secs = int(eta_secs % 60)
        eta_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        progress_bar.progress(int((idx + 1) / total * 100))
        status_text.text(f"Processing {idx+1}/{total}. ETA: {eta_str}")

        h1, headings, err = extract_h1_and_headings(url)
        if err:
            if "403" in err:
                st.error(
                    f"❌ Could not access {url}: {err}. Possible reasons: site is behind Cloudflare/WAF, IP blocked, rate-limited, or requires JS/auth."
                )
                st.session_state.skipped.append(
                    {"Address": url, "Reason": f"{err} (likely blocked/forbidden)"}
                )
            else:
                st.warning(f"⚠️ Skipped {url}: {err}.")
                st.session_state.skipped.append({"Address": url, "Reason": err})
            continue

        if not h1 and not headings:
            st.warning(f"Skipped {url}: no H1 or subheadings found.")
            st.session_state.skipped.append({"Address": url, "Reason": "No H1 or subheadings found"})
            continue

        queries = fetch_query_fan_outs(h1)
        if not queries:
            st.warning(f"Skipped {url}: no fan-out queries for H1 '{h1}'.")
            st.session_state.skipped.append(
                {"Address": url, "Reason": f"No fan-out queries returned for H1: '{h1}'"}
            )
            continue

        prompt = build_prompt(h1, headings, queries)
        results = get_explanations(prompt)
        if not results:
            st.warning(f"Skipped {url}: OpenAI returned no usable output.")
            st.session_state.skipped.append(
                {"Address": url, "Reason": "OpenAI returned no results or parsing failed"}
            )
            continue

        covered = sum(1 for it in results if it.get("covered"))
        pct = round((covered / len(results)) * 100) if results else 0
        st.session_state.summary.append({"Address": url, "Coverage (%)": pct})

        missing = [it.get("query") for it in results if not it.get("covered")]
        st.session_state.actions.append(
            {"Address": url, "Recommended Sections to Add to Content": "; ".join(missing)}
        )

        row = {
            "Address": url,
            "H1-1": h1,
            "Content Structure": " | ".join(f"{lvl}:{txt}" for lvl, txt in headings),
        }
        for i, it in enumerate(results):
            row[f"Query {i+1}"] = it.get("query", "")
            row[f"Query {i+1} Covered"] = "Yes" if it.get("covered") else "No"
            row[f"Query {i+1} Explanation"] = it.get("explanation", "")
        st.session_state.detailed.append(row)

    # Finalize
    progress_bar.progress(100)
    status_text.text("Complete!")
    st.session_state.processed = True

# Display/download results if available
if st.session_state.processed:
    st.header("Results")

    if st.session_state.detailed:
        st.subheader("Detailed")
        df_det = pd.DataFrame(st.session_state.detailed)
        st.download_button(
            "Download Detailed CSV", df_det.to_csv(index=False).encode("utf-8"), "detailed.csv", "text/csv"
        )
        st.dataframe(df_det)
    else:
        st.info("No detailed results to display.")

    if st.session_state.summary:
        st.subheader("Summary")
        df_sum = pd.DataFrame(st.session_state.summary)
        st.download_button(
            "Download Summary CSV", df_sum.to_csv(index=False).encode("utf-8"), "summary.csv", "text/csv"
        )
        st.dataframe(df_sum)
    else:
        st.info("No summary results to display.")

    if st.session_state.actions:
        st.subheader("Actions")
        df_act = pd.DataFrame(st.session_state.actions)
        st.download_button(
            "Download Actions CSV", df_act.to_csv(index=False).encode("utf-8"), "actions.csv", "text/csv"
        )
        st.dataframe(df_act)
    else:
        st.info("No actions to display.")

    if st.session_state.skipped:
        st.subheader("Skipped URLs and Reasons")
        st.table(pd.DataFrame(st.session_state.skipped))



