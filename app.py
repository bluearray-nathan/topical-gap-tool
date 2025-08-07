import subprocess
import sys
# This command ensures the browser needed by Playwright is installed.
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
    """This tool identifies where gaps exist in your content by:
1. Extracting your page's H1 and subheadings (H2–H4).
2. Using Google Gemini to generate diverse user query fan-outs (now with an optional recursive step for deeper analysis).
3. Comparing those queries against content headings to identify missing topics."""
)

# --- Settings ---
# New setting to control the fan-out method
# 0 = Original behavior (single fan-out from H1)
# 1 = Recursive (fan-out from H1, then fan-out from each of those results)
recursion_depth = 1

gemini_temp = 0.4  # fan-out diversity
gpt_temp = 0.1     # gap reasoning temperature
attempts = 2       # number of Gemini aggregation calls for the main H1
candidate_count = 5# number of candidates per Gemini call

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

# --- Helper Functions ---
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

def dedupe_queries(queries, raw_threshold=0.9, residual_threshold=0, embedding_model="text-embedding-ada-002"):
    if not queries:
        return []
    try:
        resp = openai.embeddings.create(model=embedding_model, input=queries)
        query_vecs = [np.array(d.embedding, dtype=float) for d in resp.data]
    except Exception as e:
        st.error(f"Failed to get embeddings for deduplication: {e}")
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
                        anchor_cache[anchor_text] = np.array(a_resp.data[0].embedding, dtype=float)
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

# --- CORRECTED: Helper for a single fan-out call ---
def _fetch_single_fan_out(text_to_fan_out: str, single_attempt_count: int, temp: float) -> list:
    """Makes a single call to the Gemini API to get fan-out queries for a given text."""
    queries = []
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash-latest:generateContent?key={gemini_api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": text_to_fan_out}]}],
        # THIS IS THE CORRECTED LINE
        "tools": [{"Google Search": {}}],
        "generationConfig": {
            "temperature": temp,
            "candidateCount": candidate_count
        },
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=45)
        r.raise_for_status()
        for cand in r.json().get("candidates", []):
            fanouts = cand.get("groundingMetadata", {}).get("webSearchQueries", [])
            queries.extend(fanouts)
    except Exception as e:
        # Fail silently now that the issue is understood, to keep the UI clean.
        pass
    
    time.sleep(0.5) # API rate limiting
    return list(set(queries)) # Return unique queries from this call

# --- Rewritten function for iterative fan-out ---
def fetch_query_fan_outs_multi(h1_text: str, attempts: int, temp: float, depth: int) -> list:
    """
    Generates fan-out queries. If depth > 0, performs a second level of fan-outs
    on the initial results.
    """
    st.info(f"Performing fan-out for H1: '{h1_text}' (Recursion Depth: {depth})")
    
    # Step 1: Initial Fan-Out from H1
    initial_queries = []
    for _ in range(attempts):
        initial_queries.extend(_fetch_single_fan_out(h1_text, 1, temp))
    initial_queries = list(set(initial_queries)) # Basic dedupe

    all_queries = list(initial_queries)

    # Step 2: Recursive Fan-Out (if depth > 0)
    if depth > 0 and initial_queries:
        sub_query_progress = st.status(f"Performing recursive fan-out for {len(initial_queries)} sub-queries...", expanded=False)
        processed_sub_queries = set()

        for i, query in enumerate(initial_queries):
            if query in processed_sub_queries:
                continue
            
            sub_query_progress.write(f"Level 2 Fan-out ({i+1}/{len(initial_queries)}): '{query}'")
            recursive_queries = _fetch_single_fan_out(query, single_attempt_count=1, temp=temp)
            all_queries.extend(recursive_queries)
            processed_sub_queries.add(query)

        sub_query_progress.update(label="Recursive fan-out complete!", state="complete", expanded=False)

    # Step 3: Final Comprehensive Deduplication
    seen = set()
    unique_raw = []
    for q in all_queries:
        if q not in seen:
            seen.add(q)
            unique_raw.append(q)
    
    seen_canon = set()
    filtered = []
    for q in unique_raw:
        canon = canonicalize_query(q)
        if canon not in seen_canon:
            seen_canon.add(canon)
            filtered.append(q)
            
    st.info(f"Found {len(filtered)} unique queries. Now running semantic deduplication...")
    final_queries = dedupe_queries(filtered)
    st.success(f"Deduplication complete. Final unique query count: {len(final_queries)}")
    return final_queries


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
            # Using JSON mode for more reliable output
            resp = openai.chat.completions.create(
                model="gpt-4o", 
                messages=messages, 
                temperature=temperature, 
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            parsed_json = json.loads(content)
            
            for key, value in parsed_json.items():
                if isinstance(value, list):
                    return value

            if isinstance(parsed_json, list):
                return parsed_json

        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            last_raw = f"JSON parse error: {e}\nRaw output: {content if 'content' in locals() else 'N/A'}"
        except Exception as e:
            last_raw = f"API error: {e}"
        
        if attempt < max_retries:
            messages.append({"role": "assistant", "content": content if 'content' in locals() else last_raw})
            messages.append({
                "role": "user", "content": (
                    "Your previous response was not a valid JSON object containing an array. Please try again. "
                    "Ensure the output is a single JSON object, with a key whose value is the array of results. "
                    "For example: {\"results\": [{\"query\":\"...\",\"covered\":true,\"explanation\":\"...\"}]}"
                ),
            })
            time.sleep(1 * attempt)

    st.warning(f"OpenAI parse failure after {max_retries} attempts. Raw output:\n{last_raw}")
    return []

# --- Main audit loop (only runs when user clicks) ---
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
                page.goto(url, timeout=60000, wait_until="domcontentloaded")
                
                page_title = page.title().lower()
                if "403 forbidden" in page_title or "access denied" in page_title:
                     browser.close()
                     return "", [], "HTTP 403 Forbidden (Playwright)"

                html = page.content()
                browser.close()
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            try:
                scraper = cloudscraper.create_scraper(
                    browser={"browser": "chrome", "platform": "windows", "mobile": False}
                )
                r = scraper.get(url, timeout=30)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
            except requests.exceptions.HTTPError as he:
                code = he.response.status_code if he.response else "unknown"
                return "", [], f"HTTP {code} Error (Fallback)"
            except Exception as final_e:
                return "", [], f"Fetch failed: {final_e}"
        
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        if not h1:
            h1 = soup.find("title").get_text(strip=True) if soup.find("title") else ""
            
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
        lines.extend(["", "2) User queries to check for coverage:"])
        for q in queries:
            lines.append(f"- {q}")
        lines.extend([
            "", "3) Return a JSON object where a single key contains an array of objects. Each object must have keys: query, covered, explanation.",
            'Example: {"results": [{"query":"...","covered":true,"explanation":"..."}]}',
        ])
        return "\n".join(lines)

    for idx, url in enumerate(urls):
        elapsed = time.time() - start_time
        avg_time_per_url = elapsed / (idx + 1) if idx > 0 else elapsed
        remaining_urls = total - (idx + 1)
        eta_secs = remaining_urls * avg_time_per_url
        eta_str = time.strftime("%M:%S", time.gmtime(eta_secs))
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Processing {idx+1}/{total} ({url}). ETA: {eta_str}")

        h1, headings, err = extract_h1_and_headings(url)
        if err:
            st.warning(f"⚠️ Skipped {url}: {err}.")
            st.session_state.skipped.append({"Address": url, "Reason": err})
            continue

        if not h1:
            st.warning(f"Skipped {url}: no H1 or Title found.")
            st.session_state.skipped.append({"Address": url, "Reason": "No H1 or Title tag found"})
            continue
        
        if h1 in st.session_state.h1_fanout_cache:
            fanouts = st.session_state.h1_fanout_cache[h1]
            st.info(f"Using cached fan-out queries for H1: '{h1}'")
        else:
            fanouts = fetch_query_fan_outs_multi(h1, attempts=attempts, temp=gemini_temp, depth=recursion_depth)
            st.session_state.h1_fanout_cache[h1] = fanouts
        
        if not fanouts:
            st.warning(f"Skipped {url}: no fan-out queries generated for H1 '{h1}'.")
            st.session_state.skipped.append({"Address": url, "Reason": f"No fan-out queries for H1: '{h1}'"})
            continue

        prompt = build_prompt(h1, headings, fanouts)
        results = get_explanations(prompt, temperature=gpt_temp)
        
        if not results:
            st.warning(f"Skipped {url}: OpenAI returned no usable output.")
            st.session_state.skipped.append({"Address": url, "Reason": "OpenAI parsing failed"})
            continue

        covered_count = sum(1 for it in results if it.get("covered"))
        coverage_pct = round((covered_count / len(results)) * 100) if results else 0
        
        st.session_state.summary.append({
            "Address": url, "Fan-out Count": len(fanouts), "Coverage (%)": coverage_pct
        })
        
        missing_queries = [it.get("query") for it in results if not it.get("covered")]
        st.session_state.actions.append({
            "Address": url, "Recommended Sections to Add to Content": "; ".join(missing_queries)
        })

        detailed_row = {
            "Address": url,
            "H1-1": h1,
            "Content Structure": " | ".join(f"{lvl}:{txt}" for lvl, txt in headings),
        }
        for i, it in enumerate(results):
            detailed_row[f"Query {i+1}"] = it.get("query", "")
            detailed_row[f"Query {i+1} Covered"] = "Yes" if it.get("covered") else "No"
            detailed_row[f"Query {i+1} Explanation"] = it.get("explanation", "")
        st.session_state.detailed.append(detailed_row)

    progress_bar.progress(1.0)
    status_text.text("Audit Complete!")
    st.session_state.processed = True

# --- Display / download final outputs ---
if st.session_state.processed:
    st.header("Results")

    if st.session_state.summary:
        st.subheader("Summary")
        df_sum = pd.DataFrame(st.session_state.summary)
        st.download_button(
            "Download Summary CSV", df_sum.to_csv(index=False).encode("utf-8"), "summary.csv", "text/csv", key="sum_dl"
        )
        st.dataframe(df_sum)
    
    if st.session_state.actions:
        st.subheader("Actions")
        df_act = pd.DataFrame(st.session_state.actions)
        st.download_button(
            "Download Actions CSV", df_act.to_csv(index=False).encode("utf-8"), "actions.csv", "text/csv", key="act_dl"
        )
        st.dataframe(df_act)
        
    if st.session_state.detailed:
        st.subheader("Detailed Breakdown")
        df_det = pd.DataFrame(st.session_state.detailed)
        st.download_button(
            "Download Detailed CSV", df_det.to_csv(index=False).encode("utf-8"), "detailed.csv", "text/csv", key="det_dl"
        )
        st.dataframe(df_det)

    if st.session_state.skipped:
        st.subheader("Skipped URLs and Reasons")
        st.table(pd.DataFrame(st.session_state.skipped))





