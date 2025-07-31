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
2. Using Google Gemini to generate relevant user queries (fan‑outs).
3. Comparing queries against your headings with OpenAI GPT to identify missing topics."""
)

# Load API keys from Streamlit secrets
openai.api_key    = st.secrets["openai"]["api_key"]
gemini_api_key    = st.secrets["google"]["gemini_api_key"]

# Input URLs directly
urls_input = st.text_area(
    "Enter one URL per line to analyze headers and content gaps:",
    placeholder="https://example.com/page1
https://example.com/page2"
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
    unsafe_allow_html=True
)

if urls_input:
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    total = len(urls)
    st.write(f"Found {total} URLs to process.")

    # Start button to kick off processing
    if st.button("Start Audit"):
        # Initialize progress and timer
        progress_bar = st.progress(0)
        status_text  = st.empty()
        start_time   = time.time()

        # Prepare outputs
        detailed = []
        summary  = []
        actions  = []  # recommended missing queries

        # Processing loop
        for idx, url in enumerate(urls):
            elapsed   = time.time() - start_time
            avg       = elapsed / (idx + 1)
            remaining = total - (idx + 1)
            eta_secs  = remaining * avg
            mins      = int(eta_secs // 60)
            secs      = int(eta_secs % 60)
            eta_str   = f"{mins}m {secs}s" if mins>0 else f"{secs}s"
            progress_bar.progress(int((idx+1)/total*100))
            status_text.text(f"Processing {idx+1}/{total}. ETA: {eta_str}")

            h1, headings = extract_h1_and_headings(url)
            if not h1 and not headings:
                continue

            queries = fetch_query_fan_outs(h1)
            if not queries:
                continue
            prompt  = build_prompt(h1,headings,queries)
            results = get_explanations(prompt)

            covered = sum(1 for it in results if it.get("covered"))
            pct     = round((covered/len(results))*100) if results else 0
            summary.append({"Address":url,"Coverage (%)":pct})

            missing = [it.get("query") for it in results if not it.get("covered")]
            actions.append({"Address":url, "Recommended Sections to Add to Content": "; ".join(missing)})

            row = {"Address":url,"H1-1":h1,"Content Structure":" | ".join(f"{lvl}:{txt}" for lvl,txt in headings)}
            for i, it in enumerate(results):
                row[f"Query {i+1}"]             = it.get("query","")
                row[f"Query {i+1} Covered"]     = "Yes" if it.get("covered") else "No"
                row[f"Query {i+1} Explanation"] = it.get("explanation","")
            detailed.append(row)

        # Finalize
        progress_bar.progress(100)
        status_text.text("Complete!")

        # Download outputs
        if detailed:
            df_det = pd.DataFrame(detailed)
            st.download_button("Download Detailed CSV", df_det.to_csv(index=False).encode('utf-8'), 'detailed.csv','text/csv')
            st.dataframe(df_det)
        else:
            st.info("No detailed results to display.")

        if summary:
            df_sum = pd.DataFrame(summary)
            st.download_button("Download Summary CSV", df_sum.to_csv(index=False).encode('utf-8'), 'summary.csv','text/csv')
            st.dataframe(df_sum)
        else:
            st.info("No summary results to display.")

        if actions:
            df_act = pd.DataFrame(actions)
            st.download_button("Download Actions CSV", df_act.to_csv(index=False).encode('utf-8'), 'actions.csv','text/csv')
            st.dataframe(df_act)
        else:
            st.info("No actions to display.")










