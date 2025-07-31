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

# --- Streamlit UI ---
st.set_page_config(page_title="AI Overview/AI Mode query fan-out gap analysis", layout="wide")
st.title("🔍 AI Overview/AI Mode query fan-out gap analysis")

st.sidebar.header("About the query fan-out gap analysis tool")
st.sidebar.write(
    """This identify where gaps exist in your content:
1. Extracting your page's H1 and subheadings (H2–H4).
2. Using Google Gemini to generate diverse user query fan-outs.
3. Comparing those queries against content headings to identify concise missing topics."""
)

# Fixed settings (no user adjustment)
gemini_temp = 0.9  # fan-out diversity
gpt_temp = 0.1     # gap reasoning temperature
attempts = 1       # number of Gemini aggregation calls

# Load credentials from secrets
openai.api_key = st.secrets["openai"]["api_key"]
gemini_api_key = st.secrets["google"]["gemini_api_key"]

# URL input
urls_input = st.text_area(
    "Enter one URL per line to audit:",
    placeholder="https://example.com/page1\nhttps://example.com/page2"
)

# Red start button styling
st.markdown(
    """
    <style>
    div.stButton > button:first-child { background-color: #e63946; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state initialization
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

# Prepare URLs list
urls = [u.strip() for u in urls_input.splitlines() if u.strip()] if urls_input else []
total = len(urls)
if urls:
    st.write(f"Found {total} URLs to process.")

# Reset when URLs change
if urls and st.session_state.last_urls != urls:
    st.session_state.last_urls = urls
    st.session_state.processed = False
    st.session_state.detailed = []
    st.session_state.summary = []
    st.session_state.actions = []
    st.session_state.skipped = []
    st.session_state.h1_fanout_cache = {}

# Start button
start_clicked = st.button("Start Audit")
if start_clicked:
    st.session_state.processed = False

# Helpers for query normalization & dedupe
STOPWORDS = {
    "the", "and", "of", "in", "to", "a", "for", "with", "on", "about",
    "vs", "vs.", "is", "are", "your", "what", "how", "why", "more",
    "latest", "new"
}

def canonicalize_query(q: str) -> str:
    q_lower = q.lower()
    q_lower = re.sub(r"\b(what|how|does|do|is|are|the|a|an|of|for|to|about|your)\b", " ", q_lower)
    q_lower = re.sub(r"\b(includes|including|inclusions)\b", " include ", q_lower)
    q_lower = re.sub(r"\b(works|working)\b", " work ", q_lower)
    q_lower = re.sub(r"[^\w\s]", "", q_lower)
    q_lower = re.sub(r"\s+", " ", q_lower).strip()
    return q_lower

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
        return queries
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

def fetch_query_fan_outs_multi(h1_text, attempts=3, temp=0.0):
    aggregated = []
    for attempt_i in range(attempts):
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
            st.warning(f"Fan-out fetch attempt {attempt_i+1} failed: {e}")
        time.sleep(0.2)
    # exact dedupe preserving order
    seen = set()
    unique_raw = []
    for q in aggregated:
        if q not in seen:
            seen.add(q)
            unique_raw.append(q)
    # canonical dedupe
    seen_canon = set()
    filtered = []
    for q in unique_raw:
        canon = canonicalize_query(q)
        if canon in seen_canon:
            continue
        seen_canon.add(canon)
        filtered.append(q)
    # semantic dedupe after canonical collapse
    return dedupe_queries(filtered)

def get_explanations(prompt, temperature=0.1, max_retries=2):
    system_msg = (
        "You are an SEO content gap auditor. Given the input, respond ONLY with a valid JSON array and nothing else. "
        "Each item must have exactly these keys: query (string), covered (true/false), and explanation (concise string). "
        "Do not include any extra prose. Example: [{\"query\":\"...\",\"covered\":true,\"explanation\":\"...\"}]"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=temperature, max_tokens=1000
            )
            choice = resp.choices[0]
            text = (
                choice.message.content.strip()
                if hasattr(choice, "message")
                else choice.text.strip()
            )
            last_raw = text
            cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            m = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user", "content": (
                        "Previous response was not valid JSON. Please output only the JSON array exactly as specified. "
                        "Example: [{\"query\":\"...\",\"covered\":true,\"explanation\":\"...\"}]"
                    ),
                })
        except Exception as e:
            last_raw = f"API error: {e}"
            if attempt < max_retries:
                time.sleep(1 * attempt)
                continue
    st.warning(f"OpenAI parse failure after {max_retries} attempts. Raw output:\n{last_raw}")
    return []

# Main audit loop (only runs when user clicks)
if start_clicked and urls and not st.session_state.processed:
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    st.session_state.detailed = []
    st.session_state.summary = []
    st.session_state.actions = []
    st.session_state.skipped = []

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
        lines.extend([
            "", "3) Return JSON array of objects with keys: query, covered, explanation.",
            'Example: [{"query":"...","covered":true,"explanation":"..."}]',
        ])
        return "\n".join(lines)

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
        if h1 in st.session_state.h1_fanout_cache:
            fanouts = st.session_state.h1_fanout_cache[h1]
        else:
            fanouts = fetch_query_fan_outs_multi(h1, attempts=attempts, temp=gemini_temp)
            st.session_state.h1_fanout_cache[h1] = fanouts
        if not fanouts:
            st.warning(f"Skipped {url}: no fan-out queries for H1 '{h1}'.")
            st.session_state.skipped.append(
                {"Address": url, "Reason": f"No fan-out queries returned for H1: '{h1}'"}
            )
            continue
        prompt = build_prompt(h1, headings, fanouts)
        results = get_explanations(prompt, temperature=gpt_temp)
        if not results:
            st.warning(f"Skipped {url}: OpenAI returned no usable output.")
            st.session_state.skipped.append(
                {"Address": url, "Reason": "OpenAI returned no results or parsing failed"}
            )
            continue
        covered = sum(1 for it in results if it.get("covered"))
        pct = round((covered / len(results)) * 100) if results else 0
        st.session_state.summary.append(
            {"Address": url, "Fan-out Count": len(fanouts), "Coverage (%)": pct}
        )
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

# Display / download final outputs
if st.session_state.processed:
    st.header("Results")

    if st.session_state.detailed:
        st.subheader("Detailed")
        df_det = pd.DataFrame(st.session_state.detailed)
        st.download_button(
            "Download Detailed CSV", df_det.to_csv(index=False).encode("utf-8"), "detailed.csv", "text/csv"
        )
        st.dataframe(df_det)

    if st.session_state.summary:
        st.subheader("Summary")
        df_sum = pd.DataFrame(st.session_state.summary)
        base_cols = ["Address", "Fan-out Count", "Coverage (%)"]
        ordered = [c for c in base_cols if c in df_sum.columns] + [c for c in df_sum.columns if c not in base_cols]
        df_sum = df_sum[ordered]
        st.download_button(
            "Download Summary CSV", df_sum.to_csv(index=False).encode("utf-8"), "summary.csv", "text/csv"
        )
        st.dataframe(df_sum)

    if st.session_state.actions:
        st.subheader("Actions")
        df_act = pd.DataFrame(st.session_state.actions)
        st.download_button(
            "Download Actions CSV", df_act.to_csv(index=False).encode("utf-8"), "actions.csv", "text/csv"
        )
        st.dataframe(df_act)

    if st.session_state.skipped:
        st.subheader("Skipped URLs and Reasons")
        st.table(pd.DataFrame(st.session_state.skipped))






