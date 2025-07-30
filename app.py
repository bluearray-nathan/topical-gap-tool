# app.py

import streamlit as st
import pandas as pd
import requests
import re
import json
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import openai

# --- Streamlit UI ---
st.set_page_config(page_title="Content Gap Audit", layout="wide")
st.title("🔍 Content Gap Audit Tool")

# Load API keys from Streamlit secrets
openai.api_key      = st.secrets["openai"]["api_key"]
gemini_api_key      = st.secrets["google"]["gemini_api_key"]

# CSV upload
df_file = st.file_uploader("Upload Screaming Frog CSV", type=["csv"])

if df_file:
    df = pd.read_csv(df_file)
    urls = df['Address'].dropna().unique()
    st.write(f"Found {len(urls)} URLs to process.")

    # Helper: extract H1 + headings via Playwright
    def extract_h1_and_headings(url):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)
            html = page.content()
            browser.close()
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        headings = [
            (tag.name.upper(), tag.get_text(strip=True))
            for tag in soup.find_all(['h2','h3','h4'])
        ]
        return h1, headings

    # Helper: fetch query fan‑outs via Google Gemini
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
            st.warning(f"Fan‑out fetch failed: {e}")
            return []

    # Build the GPT prompt asking for JSON output
    def build_prompt(h1, headings, queries):
        lines = [
            "I’m auditing this page for content gaps.",
            f"Main topic (H1): {h1}",
            "",
            "1) Existing headings (in order):"
        ]
        for lvl, txt in headings:
            lines.append(f"{lvl}: {txt}")
        lines.append("")
        lines.append("2) User queries to cover:")
        for q in queries:
            lines.append(f"- {q}")
        lines.append("")
        lines.append("3) Return a JSON array of objects with keys:")
        lines.append("   query: original query,")
        lines.append("   covered: true/false,")
        lines.append("   explanation: concise rationale.")
        lines.append("")
        lines.append("Example:")
        lines.append('[{"query":"...","covered":true,"explanation":"..."}]')
        return "\n".join(lines)

    # Call OpenAI and parse JSON
    def get_explanations(prompt):
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        txt = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
        try:
            arr = json.loads(txt)
            return arr if isinstance(arr, list) else []
        except:
            return []

    # Process each URL and build rows
    output = []
    for url in urls:
        st.write(f"Processing: {url}")
        h1, headings = extract_h1_and_headings(url)
        queries = fetch_query_fan_outs(h1)
        if not queries:
            continue
        prompt = build_prompt(h1, headings, queries)
        results = get_explanations(prompt)

        row = {
            "Address": url,
            "H1-1": h1,
            "Content Structure": " | ".join(f"{lvl}:{txt}" for lvl, txt in headings)
        }
        for i, it in enumerate(results):
            row[f"Query {i+1}"]             = it.get("query", "")
            row[f"Query {i+1} Covered"]     = "Yes" if it.get("covered") else "No"
            row[f"Query {i+1} Explanation"] = it.get("explanation", "")
        output.append(row)

    if output:
        df_out = pd.DataFrame(output)
        # Download button and table
        st.download_button(
            "Download CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            "content_gap_audit.csv",
            "text/csv"
        )
        st.dataframe(df_out)
