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
import numpy as np
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
3. Pre-matching those queries semantically to existing headings.
4. Using OpenAI GPT to adjudicate uncertain/missing queries and suggest headings.
5. Producing nuanced coverage scores and actionable gap reports."""
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

# Session state init
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

# Reset when input changes
if urls and st.session_state.last_urls != urls:
    st.session_state.last_urls = urls
    st.session_state.processed = False
    st.session_state.detailed = []
    st.session_state.summary = []
    st.session_state.actions = []
    st.session_state.skipped = []

# Trigger
start_clicked = st.button("Start Audit", key="start")
if start_clicked and urls:
    st.session_state.processed = False  # force re-run

# Embedding thresholds
COVERED_THRESH = 0.8
UNCERTAIN_THRESH = 0.6

# Cosine similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# Embedding retrieval
def get_embeddings(texts, model="text-embedding-ada-002"):
    try:
        resp = openai.embeddings.create(model=model, input=texts)
        return [np.array(item["embedding"], dtype=float) for item in resp["data"]]
    except Exception as e:
        st.warning(f"Embedding call failed: {e}")
        return [None] * len(texts)

# Only run processing once per URL list
if urls and not st.session_state.processed:
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    st.session_state.detailed = []
    st.session_state.summary = []
    st.session_state.actions = []
    st.session_state.skipped = []

    # Extract H1 and headings with fallback
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

    # Gemini fan-out with retries + union/dedupe
    def fetch_query_fan_outs(h1_text, tries=3, cap=12):
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        aggregated = []
        seen = set()
        backoff = 1
        for attempt in range(tries):
            payload = {
                "contents": [{"parts": [{"text": h1_text}]}],
                "tools": [{"google_search": {}}],
                "generationConfig": {"temperature": 0.0},
            }
            try:
                r = requests.post(endpoint, json=payload, timeout=30)
                r.raise_for_status()
                cand = r.json().get("candidates", [{}])[0]
                fanouts = cand.get("groundingMetadata", {}).get("webSearchQueries", []) or []
                for q in fanouts:
                    if q not in seen:
                        seen.add(q)
                        aggregated.append(q)
                if len(aggregated) >= cap:
                    break
            except Exception as e:
                st.warning(f"Fan-out attempt {attempt+1} failed: {e}")
            time.sleep(backoff)
            backoff *= 2
        return aggregated[:cap]

    # Build prompt including prelabels
    def build_prompt_with_prelabelling(h1, headings, prelabels):
        lines = [
            "I’m auditing this page for content gaps.",
            f"Main topic (H1): {h1}",
            "",
            "1) Existing headings (in order):",
        ]
        for lvl, txt in headings:
            lines.append(f"{lvl}: {txt}")
        lines.append("")
        lines.append("2) User search queries with preliminary semantic coverage (prelabel, best match heading, similarity):")
        for p in prelabels:
            lines.append(
                json.dumps(
                    {
                        "query": p["query"],
                        "prelabel": p["prelabel"],
                        "best_match_heading": p.get("best_match_heading", ""),
                        "similarity": round(p.get("similarity", 2), 2),
                    }
                )
            )
        lines.extend(
            [
                "",
                "3) Only consider queries with prelabel 'uncertain' or 'missing'.",
                "For each such query, decide if it's truly missing or sufficiently covered.",
                "Return a JSON array of objects. Each object must have:",
                '  query: original query,',
                '  covered: true/false (after review),',
                '  explanation: concise reason if missing or why acceptable,',
                '  suggested_heading: if missing, a proposed heading title (empty if covered),',
                '  confidence: one of high, medium, low indicating how clear the gap is.',
                "Example:",
                '[', 
                '{"query":"lifecycle environmental impact comparison EV vs gasoline","covered":false,"explanation":"No section ties all lifecycle stages together.","suggested_heading":"EV vs Gasoline Cars: Full Lifecycle Environmental Impact","confidence":"high"},',
                '{"query":"battery disposal environmental issues","covered":true,"explanation":"Recycling section partially covers this.","suggested_heading":"","confidence":"medium"}',
                ']',
            ]
        )
        return "\n".join(lines)

    # Call OpenAI to adjudicate gaps
    def get_gap_assessments(prompt):
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.1
            )
            txt = resp.choices[0].message.content.strip()
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)
            try:
                arr = json.loads(txt)
                return arr if isinstance(arr, list) else []
            except Exception:
                st.warning(f"Failed to parse GPT output as JSON. Raw: {txt}")
                return []
        except Exception as e:
            st.warning(f"OpenAI call failed: {e}")
            return []

    # Process each URL
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

        fanouts = fetch_query_fan_outs(h1)
        if not fanouts:
            st.warning(f"Skipped {url}: no fan-out queries for H1 '{h1}'.")
            st.session_state.skipped.append(
                {"Address": url, "Reason": f"No fan-out queries returned for H1: '{h1}'"}
            )
            continue

        # Embedding pre-match
        heading_texts = [f"{lvl}: {txt}" for lvl, txt in headings]
        heading_embs = get_embeddings(heading_texts)
        query_embs = get_embeddings(fanouts)
        prelabels = []
        for qi, q in enumerate(fanouts):
            sims = []
            for hi, h_emb in enumerate(heading_embs):
                sims.append(cosine_similarity(query_embs[qi], h_emb) if query_embs[qi] is not None else 0.0)
            best_sim = max(sims) if sims else 0.0
            best_heading = heading_texts[sims.index(best_sim)] if sims else ""
            if best_sim >= COVERED_THRESH:
                label = "covered"
            elif best_sim >= UNCERTAIN_THRESH:
                label = "uncertain"
            else:
                label = "missing"
            prelabels.append(
                {
                    "query": q,
                    "best_match_heading": best_heading,
                    "similarity": best_sim,
                    "prelabel": label,
                }
            )

        # Determine which to send to GPT
        to_review = [p for p in prelabels if p["prelabel"] in ("uncertain", "missing")]
        gpt_results = []
        if to_review:
            prompt = build_prompt_with_prelabelling(h1, headings, to_review)
            gpt_results = get_gap_assessments(prompt)

        # Combine signals and compute weights
        detailed_rows_for_url = []
        total_weight = 0.0
        for p in prelabels:
            q = p["query"]
            prelabel = p["prelabel"]
            best_heading = p.get("best_match_heading", "")
            similarity = p.get("similarity", 0.0)
            gpt_entry = next((g for g in gpt_results if g.get("query", "") == q), None)

            gap_flagged = False
            explanation = ""
            suggested_heading = ""
            confidence = ""
            final_weight = 0.0
            final_covered = False
            gpt_covered = True  # default assume covered if no entry

            if prelabel == "covered":
                # base: fully covered
                final_weight = 1.0
                final_covered = True
                explanation = "Pre-match semantic similarity indicates strong coverage."
                confidence = "high"
                if gpt_entry and not gpt_entry.get("covered", True):
                    gpt_covered = False
                    gap_flagged = True
                    explanation = gpt_entry.get("explanation", "")
                    suggested_heading = gpt_entry.get("suggested_heading", "")
                    confidence = gpt_entry.get("confidence", "medium")
                    if confidence == "high":
                        final_weight = 0.5
                    elif confidence == "medium":
                        final_weight = 0.75
                    else:
                        final_weight = 0.9
                    final_covered = final_weight >= 0.75
            else:
                # uncertain or missing
                if gpt_entry:
                    gpt_covered = bool(gpt_entry.get("covered", False))
                    confidence = gpt_entry.get("confidence", "medium")
                    explanation = gpt_entry.get("explanation", "")
                    suggested_heading = gpt_entry.get("suggested_heading", "")
                    if gpt_entry.get("covered", False):
                        final_weight = 1.0
                        final_covered = True
                    else:
                        gap_flagged = True
                        if prelabel == "uncertain":
                            if confidence == "high":
                                final_weight = 0.0
                            elif confidence == "medium":
                                final_weight = 0.3
                            else:
                                final_weight = 0.5
                        else:  # missing
                            if confidence == "high":
                                final_weight = 0.0
                            elif confidence == "medium":
                                final_weight = 0.3
                            else:
                                final_weight = 0.5
                        final_covered = final_weight >= 0.75
                else:
                    # No GPT feedback: fallback heuristics
                    if prelabel == "uncertain":
                        final_weight = 1.0
                        final_covered = True
                        explanation = "No GPT contradiction; treating uncertain as covered."
                        confidence = "medium"
                    else:  # missing
                        final_weight = 0.0
                        final_covered = False
                        explanation = "No GPT response; treated as missing."
                        confidence = "high"
                        gpt_covered = False

            total_weight += final_weight
            row = {
                "Address": url,
                "H1-1": h1,
                "Content Structure": " | ".join(f"{lvl}:{txt}" for lvl, txt in headings),
                "Query": q,
                "Prelabel": prelabel,
                "Best Match Heading": best_heading,
                "Similarity": round(similarity, 3),
                "GPT Covered": gpt_covered,
                "GPT Explanation": explanation,
                "Suggested Heading": suggested_heading,
                "Confidence": confidence,
                "Final Weight": final_weight,
                "Final Covered": final_covered,
            }
            st.session_state.detailed.append(row)
            detailed_rows_for_url.append(row)

        # Summary aggregation
        num_queries = len(fanouts)
        coverage_score = round((total_weight / num_queries) * 100, 1) if num_queries else 0
        high_conf_gaps = sum(
            1
            for r in detailed_rows_for_url
            if not r["Final Covered"] and r["Confidence"] == "high"
        )
        lower_conf_gaps = sum(
            1
            for r in detailed_rows_for_url
            if not r["Final Covered"] and r["Confidence"] in ("medium", "low")
        )
        suggested = list({r["Suggested Heading"] for r in detailed_rows_for_url if r["Suggested Heading"]})
        st.session_state.summary.append(
            {
                "Address": url,
                "Fan-out Count": num_queries,
                "Coverage (%)": coverage_score,
                "High-confidence gaps": high_conf_gaps,
                "Lower-confidence gaps": lower_conf_gaps,
                "Suggested Headings": "; ".join(suggested),
            }
        )

        # Actions: missing queries with their suggested headings
        action_suggestions = []
        for r in detailed_rows_for_url:
            if not r["Final Covered"] and r["Suggested Heading"]:
                action_suggestions.append(f"{r['Query']} -> {r['Suggested Heading']}")
        st.session_state.actions.append(
            {
                "Address": url,
                "Recommended Sections to Add to Content": "; ".join(action_suggestions),
            }
        )

    # Mark done
    progress_bar.progress(100)
    status_text.text("Complete!")
    st.session_state.processed = True

# Display / download results
if st.session_state.processed:
    st.header("Results")

    if st.session_state.detailed:
        st.subheader("Detailed per-query breakdown")
        df_det = pd.DataFrame(st.session_state.detailed)
        st.download_button(
            "Download Detailed CSV",
            df_det.to_csv(index=False).encode("utf-8"),
            "detailed.csv",
            "text/csv",
        )
        st.dataframe(df_det)
    else:
        st.info("No detailed results to display.")

    if st.session_state.summary:
        st.subheader("Summary")
        df_sum = pd.DataFrame(st.session_state.summary)
        base_cols = ["Address", "Fan-out Count", "Coverage (%)", "High-confidence gaps", "Lower-confidence gaps", "Suggested Headings"]
        ordered = [c for c in base_cols if c in df_sum.columns] + [c for c in df_sum.columns if c not in base_cols]
        df_sum = df_sum[ordered]
        st.download_button(
            "Download Summary CSV",
            df_sum.to_csv(index=False).encode("utf-8"),
            "summary.csv",
            "text/csv",
        )
        st.dataframe(df_sum)
    else:
        st.info("No summary results to display.")

    if st.session_state.actions:
        st.subheader("Actions")
        df_act = pd.DataFrame(st.session_state.actions)
        st.download_button(
            "Download Actions CSV",
            df_act.to_csv(index=False).encode("utf-8"),
            "actions.csv",
            "text/csv",
        )
        st.dataframe(df_act)
    else:
        st.info("No actions to display.")

    if st.session_state.skipped:
        st.subheader("Skipped URLs and Reasons")
        st.table(pd.DataFrame(st.session_state.skipped))

