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
st.set_page_config(page_title="AI Overview/AI Mode query fan-out gap analysis", layout="wide")
st.title("🔍 AI Overview/AI Mode query fan-out gap analysis")

# Sidebar explanation
st.sidebar.header("About AI Overview/AI Mode query fan-out gap analysis")
st.sidebar.write(
    """This identify where gaps exist in your content in relation to AI Overview/AI Mode query fan-outs:
1. Extracting your page's H1 and subheadings (H2–H4).
2. Using Google Gemini to generate relevant query fan-outs.
3. Comparing queries against your headings to identify missing topics."""
)

# Load API keys
openai.api_key    = st.secrets["openai"]["api_key"]
gemini_api_key    = st.secrets["google"]["gemini_api_key"]

# Input URLs directly
urls_input = st.text_area(
    "Enter one URL per line to analyse your content and discover gaps:",
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
    unsafe_allow_html=True
)

if urls_input:
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    total = len(urls)
    st.write(f"Found {total} URLs to process.")
    
    if st.button("Start Audit"):
        # Initialize
        progress_bar = st.progress(0)
        status_text  = st.empty()
        start_time   = time.time()
        
        # Prepare outputs
        detailed = []
        summary  = []
        actions  = []
        
        # Helpers
        def extract_h1_and_headings(url):
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, timeout=60000)
                    html = page.content()
                    browser.close()
                soup = BeautifulSoup(html, "html.parser")
            except Exception:
                try:
                    scraper = cloudscraper.create_scraper(
                        browser={"browser":"chrome","platform":"windows","mobile":False}
                    )
                    resp = scraper.get(url, timeout=30)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "html.parser")
                except requests.exceptions.HTTPError as he:
                    # Bubble up status code for main loop
                    return None, he.response.status_code
                except Exception:
                    return "", []
            h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
            headings = [(tag.name.upper(), tag.get_text(strip=True)) for tag in soup.find_all(["h2","h3","h4"])]
            return h1, headings
        
        def fetch_query_fan_outs(h1_text):
            endpoint = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
            )
            payload = {
                "contents": [{"parts": [{"text": h1_text}]}],
                "tools":    [{"google_search": {}}],
                "generationConfig": {"temperature": 1.0}
            }
            try:
                r = requests.post(endpoint, json=payload, timeout=30)
                r.raise_for_status()
                cand = r.json().get("candidates", [{}])[0]
                return cand.get("groundingMetadata", {}).get("webSearchQueries", [])
            except Exception as e:
                st.warning(f"Fan-out fetch failed: {e}")
                return []
        
        def build_prompt(h1, headings, queries):
            lines = [
                "I’m auditing this page for content gaps.",
                f"Main topic (H1): {h1}",
                "", "1) Existing headings (in order):"
            ]
            for lvl, txt in headings:
                lines.append(f"{lvl}: {txt}")
            lines.extend(["", "2) User queries to cover:"])
            for q in queries:
                lines.append(f"- {q}")
            lines.extend([
                "", 
                "3) Return JSON array of objects with keys: query, covered, explanation.",
                "Example: [{\"query\":\"...\",\"covered\":true,\"explanation\":\"...\"}]"
            ])
            return "\n".join(lines)
        
        def get_explanations(prompt):
            resp = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2
            )
            txt = resp.choices[0].message.content.strip()
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)
            try:
                arr = json.loads(txt)
                return arr if isinstance(arr, list) else []
            except:
                return []
        
        # Processing
        for idx, url in enumerate(urls):
            elapsed   = time.time() - start_time
            avg       = elapsed / (idx + 1)
            remaining = total - (idx + 1)
            eta_secs  = remaining * avg
            mins      = int(eta_secs // 60)
            secs      = int(eta_secs % 60)
            eta_str   = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            progress_bar.progress(int((idx+1)/total*100))
            status_text.text(f"Processing {idx+1}/{total}. ETA: {eta_str}")

            result = extract_h1_and_headings(url)
            # Handle cloudflare 403 bubble-up
            if isinstance(result[1], int) and result[1] == 403:
                st.error(
                    f"❌ Could not access {url} (HTTP 403 Forbidden). "
                    "Possible reasons: site is behind Cloudflare/WAF, IP blocked, or incorrect URL."
                )
                continue
            h1, headings = result
            if not h1 and not headings:
                st.error(
                    f"❌ Could not fetch content for {url}. "
                    "Possible reasons: network error, invalid URL, or site block."
                )
                continue

            queries = fetch_query_fan_outs(h1)
            if not queries:
                continue
            prompt  = build_prompt(h1, headings, queries)
            results = get_explanations(prompt)

            covered = sum(1 for it in results if it.get("covered"))
            pct     = round((covered / len(results)) * 100) if results else 0
            summary.append({"Address": url, "Coverage (%)": pct})
            missing = [it.get("query") for it in results if not it.get("covered")]
            actions.append({"Address": url, "Recommended Sections to Add to Content": "; ".join(missing)})

            row = {
                "Address": url,
                "H1-1": h1,
                "Content Structure": " | ".join(f"{lvl}:{txt}" for lvl, txt in headings)
            }
            for i, it in enumerate(results):
                row[f"Query {i+1}"]             = it.get("query", "")
                row[f"Query {i+1} Covered"]     = "Yes" if it.get("covered") else "No"
                row[f"Query {i+1} Explanation"] = it.get("explanation", "")
            detailed.append(row)

        # Finalize
        progress_bar.progress(100)
        status_text.text("Complete!")

        # Outputs
        if detailed:
            df_det = pd.DataFrame(detailed)
            st.download_button(
                "Download Detailed CSV",
                df_det.to_csv(index=False).encode('utf-8'),
                'detailed.csv',
                'text/csv'
            )
            st.dataframe(df_det)
        else:
            st.info("No detailed results to display.")

        if summary:
            df_sum = pd.DataFrame(summary)
            st.download_button(
                "Download Summary CSV",
                df_sum.to_csv(index=False).encode('utf-8'),
                'summary.csv',
                'text/csv'
            )
            st.dataframe(df_sum)
        else:
            st.info("No summary results to display.")

        if actions:
            df_act = pd.DataFrame(actions)
            st.download_button(
                "Download Actions CSV",
                df_act.to_csv(index=False).encode('utf-8'),
                'actions.csv',
                'text/csv'
            )
            st.dataframe(df_act)
        else:
            st.info("No actions to display.")







