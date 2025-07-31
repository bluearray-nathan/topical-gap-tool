import subprocess
import sys
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)

import time
import streamlit as st
import pandas as pd
import requests
import re
import json
import numpy as np
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import cloudscraper
import openai

# UI setup
st.set_page_config(page_title="Content Gap Audit", layout="wide")
st.title("🔍 Content Gap Audit Tool")

st.sidebar.header("About Content Gap Audit")
st.sidebar.write(
    """This tool audits content by:
1. Extracting H1/H2-H4 structure.
2. Getting query fan-outs from Gemini (multi-call + aggregate).
3. Deduplicating semantically similar fan-outs via embeddings.
4. Using OpenAI to identify gaps vs. headings."""
)

# Load keys
openai.api_key = st.secrets["openai"]["api_key"]
gemini_api_key = st.secrets["google"]["gemini_api_key"]

# Input
urls_input = st.text_area(
    "Enter one URL per line to analyze headers and content gaps:",
    placeholder="https://example.com/page1\nhttps://example.com/page2"
)

# Red button style
st.markdown(
    """
    <style>
    div.stButton > button:first-child { background-color: #e63946; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state init
for key, default in {
    "last_urls": [],
    "processed": False,
    "detailed": [],
    "summary": [],
    "actions": [],
    "skipped": [],
    "h1_fanout_cache": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Parse URLs
urls = [u.strip() for u in urls_input.splitlines() if u.strip()] if urls_input else []
total = len(urls)
if urls:
    st.write(f"Found {total} URLs to process.")

# Reset if changed
if urls and st.session_state.last_urls != urls:
    st.session_state.last_urls = urls
    st.session_state.processed = False
    st.session_state.detailed = []
    st.session_state.summary = []
    st.session_state.actions = []
    st.session_state.skipped = []
    st.session_state.h1_fanout_cache = {}

# Start trigger
start_clicked = st.button("Start Audit")
if start_clicked and urls:
    st.session_state.processed = False  # force run

# Embedding similarity helpers (short practical fix)
STOPWORDS = {
    "the", "and", "of", "in", "to", "a", "for", "with", "on", "about",
    "vs", "vs.", "is", "are", "your", "what", "how", "why", "more",
    "latest", "new"
}


def content_words(s: str):
    tokens = re.findall(r"[A-Za-z0-9]+", s.lower())
    return [t for t in tokens if t not in STOPWORDS]


def cosine(a: np.ndarray, b: np.ndarray):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def remove_component(vec: np.ndarray, anchor: np.ndarray):
    denom = np.dot(anchor, anchor)
    if denom == 0:
        return vec
    proj = (np.dot(vec, anchor) / denom) * anchor
    return vec - proj


def dedupe_queries(queries, raw_threshold=0.9, residual_threshold=0.5, embedding_model="text-embedding-ada-002"):
    if not queries:
        return []

    try:
        resp = openai.embeddings.create(model=embedding_model, input=queries)
        query_vecs = [np.array(d["embedding"], dtype=float) for d in resp["data"]]
    except Exception:
        return queries  # fallback

    kept = []
    removed = set()
    anchor_cache = {}

    for i, qi in enumerate(queries):
        if i in removed:
            continue
        kept.append(qi)
        vi = query_vecs[i]
        for j in range(i + 1, len(queries)):
            if j in removed:
                continue
            vj = query_vecs[j]
            raw_sim = cosine(vi, vj)
            if raw_sim < raw_threshold:
                continue
            shared = set(content_words(qi)) & set(content_words(queries[j]))
            if shared:
                anchor_text = " ".join(sorted(shared))
                if anchor_text not in anchor_cache:
                    try:
                        a_resp = openai.embeddings.create(model=embedding_model, input=[anchor_text])
                        anchor_cache[anchor_text] = np.array(a_resp["data"][0]["embedding"], dtype=float)
                    except Exception:
                        anchor_cache[anchor_text] = None
                anchor_vec = anchor_cache.get(anchor_text)
                if anchor_vec is not None:
                    ri = remove_component(vi, anchor_vec)
                    rj = remove_component(vj, anchor_vec)
                    residual_sim = cosine(ri, rj)
                else:
                    residual_sim = raw_sim
            else:
                residual_sim = raw_sim

            if residual_sim >= residual_threshold:
                removed.add(j)
    return kept


# Gemini multi-call + aggregate + dedupe
def fetch_query_fan_outs_multi(h1_text, attempts=3, temp=0.0):
    aggregated = []
    for attempt in range(attempts):
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": h1_text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": temp},
        }
        try:
            r = requests.post(endpoint, json=payload, timeout=30)
            r.raise_for_status()
            cand = r.json().get("candidates", [{}])[0]
            fanouts = cand.get("groundingMetadata", {}).get("webSearchQueries", [])
            aggregated.extend(fanouts)
        except Exception as e:
            st.warning(f"Fan-out fetch attempt {attempt+1} failed: {e}")
        time.sleep(0.2)
    # dedupe exact
    seen = set()
    unique_raw = []
    for q in aggregated:
        if q not in seen:
            seen.add(q)
            unique_raw.append(q)
    # semantic dedupe
    return dedupe_queries(unique_raw)


# Main processing
if urls and not st.session_state.processed:
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    # clear previous
    st.session_state.detailed = []
    st.session_state.summary = []
    st.session_state.actions = []
    st.session_state.skipped = []

    # extract headings
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
                return "", [], "Fetch failed"
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        headings = [(tag.name.upper(), tag.get_text(strip=True)) for tag in soup.find_all(["h2", "h3", "h4"])]
        return h1, headings, None

    # build prompt
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

    # call OpenAI
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

    for idx, url in enumerate(urls):
        elapsed = time.time() - start_time
        avg = elapsed / (idx + 1)
        remaining = total - (idx + 1)
        eta = remaining * avg
        mins = int(eta // 60)
        secs = int(eta % 60)
        eta_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        progress_bar.progress(int((idx + 1) / total * 100))
        status_text.text(f"Processing {idx+1}/{total}. ETA: {eta_str}")

        h1, headings, err = extract_h1_and_headings(url)
        if err:
            if "403" in err:
                st.error(f"❌ Could not access {url}: {err}. Possible reasons: WAF/Cloudflare or rate-limiting.")
                st.session_state.skipped.append({"Address": url, "Reason": f"{err} (likely blocked)"})
            else:
                st.warning(f"⚠️ Skipped {url}: {err}.")
                st.session_state.skipped.append({"Address": url, "Reason": err})
            continue

        if not h1 and not headings:
            st.warning(f"Skipped {url}: no H1 or headings found.")
            st.session_state.skipped.append({"Address": url, "Reason": "No H1 or subheadings"})
            continue

        # fan-outs with caching by H1
        if h1 in st.session_state.h1_fanout_cache:
            fanouts = st.session_state.h1_fanout_cache[h1]
        else:
            fanouts = fetch_query_fan_outs_multi(h1, attempts=3, temp=0.0)
            st.session_state.h1_fanout_cache[h1] = fanouts

        if not fanouts:
            st.warning(f"Skipped {url}: no fan-out queries.")
            st.session_state.skipped.append({"Address": url, "Reason": "No fan-outs returned"})
            continue

        prompt = build_prompt(h1, headings, fanouts)
        results = get_explanations(prompt)
        if not results:
            st.warning(f"Skipped {url}: OpenAI returned nothing/parsing failed.")
            st.session_state.skipped.append({"Address": url, "Reason": "GPT output parse failure"})
            continue

        covered = sum(1 for it in results if it.get("covered"))
        pct = round((covered / len(results)) * 100) if results else 0
        st.session_state.summary.append({"Address": url, "Fan-out Count": len(fanouts), "Coverage (%)": pct})

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

    progress_bar.progress(100)
    status_text.text("Complete!")
    st.session_state.processed = True

# Display
if st.session_state.processed:
    st.header("Results")

    if st.session_state.detailed:
        st.subheader("Detailed")
        df_det = pd.DataFrame(st.session_state.detailed)
        st.download_button("Download Detailed CSV", df_det.to_csv(index=False).encode("utf-8"), "detailed.csv", "text/csv")
        st.dataframe(df_det)

    if st.session_state.summary:
        st.subheader("Summary")
        df_sum = pd.DataFrame(st.session_state.summary)
        cols = ["Address", "Fan-out Count", "Coverage (%)"]
        ordered = [c for c in cols if c in df_sum.columns] + [c for c in df_sum.columns if c not in cols]
        df_sum = df_sum[ordered]
        st.download_button("Download Summary CSV", df_sum.to_csv(index=False).encode("utf-8"), "summary.csv", "text/csv")
        st.dataframe(df_sum)

    if st.session_state.actions:
        st.subheader("Actions")
        df_act = pd.DataFrame(st.session_state.actions)
        st.download_button("Download Actions CSV", df_act.to_csv(index=False).encode("utf-8"), "actions.csv", "text/csv")
        st.dataframe(df_act)

    if st.session_state.skipped:
        st.subheader("Skipped URLs and Reasons")
        st.table(pd.DataFrame(st.session_state.skipped))




