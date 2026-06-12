import subprocess
import sys
# Ensure Playwright is installed for browser automation
subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=False)

import csv
import time

import pandas as pd
import streamlit as st

import branding
import competitors
import config
import coverage
import entities as entity_mod
import extraction
import fanout
import text_utils

# --- Page + brand setup ---
st.set_page_config(page_title="AI fan-out gap analysis | Blue Array", layout="wide")
config.configure_apis()
branding.inject_css()
branding.render_header(
    "AI fan-out gap analysis",
    "Map your page against the entities and queries Google's AI explores, and against your competitors.",
)


# =========================
# CSV HELPERS (Google Sheets safe)
# =========================

_SHEETS_RISK = ("=", "+", "-", "@", "\t")


def _san(v):
    if pd.isna(v):
        return ""
    if isinstance(v, (int, float)):
        return v
    s = str(v)
    return "'" + s if s.startswith(_SHEETS_RISK) else s


def _safe(df):
    return df.map(_san)


def _csv_bytes(df):
    return df.to_csv(index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL).encode("utf-8")


# --- Status colouring (palette tints only) ---
def _status_css(v):
    s = str(v)
    if s == "Missing":
        return "background-color: rgba(236, 78, 100, 0.18);"   # coral tint
    if s == "Thin":
        return "background-color: rgba(242, 242, 242, 1);"      # light grey
    if s == "Strong":
        return "background-color: rgba(18, 145, 210, 0.16);"    # primary blue tint
    return ""


def _style_status(df, cols):
    existing = [c for c in cols if c in df.columns]
    return df.style.map(_status_css, subset=existing)


# =========================
# INTRO
# =========================

st.markdown(
    """
    <div class="ba-panel">
    <b>What this does</b><br>
    1. Reads the main content of each page (boilerplate, navigation and footers removed).<br>
    2. Uses Gemini with Google Search grounding to generate multi-layer fan-out queries, the way AI Overviews and AI Mode explore a topic.<br>
    3. Clusters those queries into <b>entities</b>: the named things and key subtopics a topic is built from.<br>
    4. Scores how well your page covers each entity (missing, thin or strong) with supporting evidence.<br>
    5. Optionally benchmarks your page against competitors to show which entities they cover and you do not.
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# =========================
# INPUTS
# =========================

urls_input = st.text_area(
    "Enter one URL per line to audit:",
    placeholder="https://example.com/page1\nhttps://example.com/page2",
)

target_keyword = st.text_input(
    "Target keyword (optional):",
    placeholder="e.g. smart export guarantee",
    help="Seeds the fan-out and competitor discovery. If left blank, the page H1 is used.",
)

col_a, col_b = st.columns(2)
with col_a:
    country = st.selectbox(
        "Target country:",
        config.COUNTRIES,
        index=0,
        help="Grounding searches and generated queries are focused on this country.",
    )
with col_b:
    max_queries = st.slider(
        "Max queries per URL:",
        min_value=5, max_value=40, value=15, step=1,
        help="Upper limit on fan-out queries (after dedupe) used to build the entity map. "
             "Lower gives broader, more actionable entities; higher gives more granular coverage.",
    )

urls = [line.strip() for line in urls_input.splitlines() if line.strip()]

# =========================
# COMPETITOR SETUP
# =========================

compare_competitors = st.checkbox("Benchmark against competitors", value=False)

if compare_competitors:
    st.caption(
        "Competitor benchmarking runs for the first URL in the list, using its target keyword "
        "(or H1). Each competitor adds a fetch and a scoring pass, so more competitors means a "
        "longer run."
    )
    if competitors.serpapi_available():
        if st.button("Suggest competitors from Google", key="suggest_comp"):
            primary = urls[0] if urls else ""
            seed = target_keyword.strip()
            if not seed and primary:
                with st.spinner("Reading the page to find a seed keyword..."):
                    seed = extraction.cached_extract_page(primary).h1
            if not seed:
                st.warning("Enter a target keyword or at least one URL first.")
            else:
                with st.spinner("Finding top organic results..."):
                    suggestions = competitors.discover_competitors(seed, country, primary)
                existing = [u for u in st.session_state.get("competitor_urls_text", "").splitlines() if u.strip()]
                merged = list(dict.fromkeys(existing + suggestions))
                st.session_state["competitor_urls_text"] = "\n".join(merged)
                if not suggestions:
                    st.info("No competitors returned. Check the keyword, or paste URLs manually below.")
    else:
        st.caption(
            "Add a SerpAPI key to the app secrets to enable auto-suggestions. "
            "You can still paste competitor URLs below."
        )

    st.text_area(
        "Competitor URLs (one per line):",
        key="competitor_urls_text",
        placeholder="https://competitor-a.com/page\nhttps://competitor-b.com/page",
    )

# =========================
# SESSION STATE
# =========================

def init_state():
    defaults = {
        "last_inputs": None,
        "processed": False,
        "summary": [],
        "entity_maps": [],   # list of {"url":..., "rows":[...]}
        "actions": [],
        "skipped": [],
        "raw_by_url": {},
        "primary_url": "",
        "primary_entities": [],
        "competitor_results": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()

# Reset results when the core inputs change (keep typed competitor URLs).
current_inputs = (tuple(urls), target_keyword.strip(), country, max_queries)
if st.session_state.last_inputs != current_inputs:
    st.session_state.last_inputs = current_inputs
    st.session_state.processed = False
    for key in ("summary", "entity_maps", "actions", "skipped"):
        st.session_state[key].clear()
    st.session_state.raw_by_url = {}
    st.session_state.primary_entities = []
    st.session_state.competitor_results = {}


# =========================
# AUDIT
# =========================

def _build_fanout_queries(seed):
    lvl1 = fanout.cached_fanouts(seed, config.CANDIDATE_COUNT, config.GEMINI_TEMP, config.LVL1_TIMEOUT, country)
    lvl2 = fanout.expand_level2_parallel(
        lvl1, attempts=config.ATTEMPTS, temp=config.GEMINI_TEMP,
        max_workers=config.MAX_WORKERS, cand_count=config.LVL2_CANDIDATES,
        timeout_s=config.LVL2_TIMEOUT, country=country,
    )
    all_qs = (lvl1 or []) + (lvl2 or [])
    if config.NORMALIZE_YEAR_SUFFIX:
        all_qs = [text_utils.strip_trailing_years(q) for q in all_qs]
    all_qs = [text_utils.strip_geo_tokens(q) for q in all_qs]
    all_qs = [q for q in all_qs if q]
    return all_qs


def _dedupe_and_cap(all_qs):
    raw = all_qs[:]
    if config.ENABLE_DEDUPE and all_qs:
        reps, _groups = text_utils.dedupe_pipeline(
            all_qs, use_exact=True, fuzzy_ratio=config.FUZZY_RATIO,
            use_embed=config.EMBED_ON, embed_threshold=config.EMBED_THRESHOLD,
        )
    else:
        seen, reps = set(), []
        for q in all_qs:
            if q.lower() not in seen:
                seen.add(q.lower())
                reps.append(q)
    return raw, reps[:max_queries]


start_clicked = st.button("Start audit")

if start_clicked and urls and not st.session_state.processed:
    progress = st.progress(0)
    status = st.empty()
    total = len(urls)

    for idx, url in enumerate(urls):
        status.text(f"Processing {idx + 1}/{total}: {url}")

        page = extraction.cached_extract_page(url)
        if not page.ok:
            reason = page.error or "No H1 or content found"
            st.warning(f"Skipped {url}: {reason}")
            st.session_state.skipped.append({"Address": url, "Reason": reason})
            progress.progress(int(((idx + 1) / total) * 100))
            continue

        seed = target_keyword.strip() or page.h1
        all_qs = _build_fanout_queries(seed)
        if not all_qs:
            st.warning(f"Skipped {url}: no fan-out queries generated")
            st.session_state.skipped.append({"Address": url, "Reason": "No queries generated"})
            progress.progress(int(((idx + 1) / total) * 100))
            continue

        raw_queries, used_queries = _dedupe_and_cap(all_qs)

        ents = entity_mod.cluster_entities(used_queries, topic=seed, country=country)
        if not ents:
            st.warning(f"Skipped {url}: could not cluster queries into entities")
            st.session_state.skipped.append({"Address": url, "Reason": "Entity clustering failed"})
            progress.progress(int(((idx + 1) / total) * 100))
            continue

        rows = entity_mod.build_entity_map(page, ents)

        counts = {"Strong": 0, "Thin": 0, "Missing": 0}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1

        st.session_state.summary.append({
            "Address": url,
            "Words": page.word_count,
            "Fan-out (raw)": len(raw_queries),
            "Queries used": len(used_queries),
            "Entities": len(rows),
            "Strong": counts["Strong"],
            "Thin": counts["Thin"],
            "Missing": counts["Missing"],
            "Entity coverage (%)": entity_mod.entity_coverage_percent(rows),
        })
        st.session_state.entity_maps.append({"url": url, "rows": rows})
        st.session_state.raw_by_url[url] = {"raw": raw_queries, "used": used_queries}

        gaps = [f"{r['entity']} ({r['status']})" for r in rows if r["status"] != "Strong"]
        st.session_state.actions.append({
            "Address": url,
            "Entities to add or strengthen": "; ".join(gaps),
        })

        # Keep the first URL's entities for competitor benchmarking.
        if idx == 0:
            st.session_state.primary_url = url
            st.session_state.primary_entities = ents

        progress.progress(int(((idx + 1) / total) * 100))

    # Competitor benchmarking (primary URL only)
    if compare_competitors and st.session_state.primary_entities:
        comp_urls = [u.strip() for u in st.session_state.get("competitor_urls_text", "").splitlines() if u.strip()]
        comp_urls = comp_urls[: config.MAX_COMPETITORS]
        if comp_urls:
            status.text(f"Benchmarking {len(comp_urls)} competitor(s)...")
            st.session_state.competitor_results = competitors.benchmark_competitors(
                st.session_state.primary_entities, comp_urls
            )

    status.text("Done.")
    st.session_state.processed = True


# =========================
# RESULTS
# =========================

if st.session_state.processed:
    st.header("Results")

    if st.session_state.summary:
        st.subheader("Summary")
        df_summary = _safe(pd.DataFrame(st.session_state.summary))
        st.dataframe(df_summary, use_container_width=True)
        st.download_button(
            "Download summary CSV", _csv_bytes(df_summary), "summary.csv", "text/csv"
        )

    # Entity coverage maps
    if st.session_state.entity_maps:
        st.subheader("Entity coverage map")
        st.caption("Sorted with missing and thin entities first. These are your content gaps.")

        combined = []
        for block in st.session_state.entity_maps:
            url = block["url"]
            display = pd.DataFrame([
                {
                    "Entity": r["entity"],
                    "Type": r["type"],
                    "Status": r["status"],
                    "On schema": "Yes" if r["schema_supported"] else "",
                    "Evidence": r["evidence"],
                    "Fan-out queries": "; ".join(r["queries"]),
                }
                for r in block["rows"]
            ])
            st.markdown(f"**{url}**")
            st.dataframe(_style_status(display, ["Status"]), use_container_width=True)
            for r in block["rows"]:
                combined.append({
                    "Address": url,
                    "Entity": r["entity"],
                    "Type": r["type"],
                    "Status": r["status"],
                    "On schema": "Yes" if r["schema_supported"] else "",
                    "Evidence": r["evidence"],
                    "Fan-out queries": "; ".join(r["queries"]),
                })

        df_entities = _safe(pd.DataFrame(combined))
        st.download_button(
            "Download entity coverage CSV", _csv_bytes(df_entities), "entity_coverage.csv", "text/csv"
        )

    # Actions
    if st.session_state.actions:
        st.subheader("Actions")
        df_actions = _safe(pd.DataFrame(st.session_state.actions))
        st.dataframe(df_actions, use_container_width=True)
        st.download_button(
            "Download actions CSV", _csv_bytes(df_actions), "actions.csv", "text/csv"
        )

    # Competitor benchmarking
    if st.session_state.competitor_results:
        st.subheader("Competitor benchmarking")
        primary_block = next(
            (b for b in st.session_state.entity_maps if b["url"] == st.session_state.primary_url),
            None,
        )
        results = st.session_state.competitor_results
        failed = {u: r for u, r in results.items() if not r.get("ok")}

        if primary_block:
            matrix_rows, labels = competitors.build_matrix(primary_block["rows"], results)
            if matrix_rows:
                df_matrix = pd.DataFrame(matrix_rows)
                comp_cols = ["Your page"] + list(labels.values())
                st.markdown(f"**Entity coverage: {competitors.domain_of(st.session_state.primary_url)} vs competitors**")
                st.dataframe(_style_status(_safe(df_matrix), comp_cols), use_container_width=True)
                st.download_button(
                    "Download competitor matrix CSV", _csv_bytes(_safe(df_matrix)),
                    "competitor_matrix.csv", "text/csv",
                )

            advantage = competitors.competitor_advantage(primary_block["rows"], results)
            if advantage:
                st.markdown("**Where competitors cover an entity and your page does not**")
                df_adv = _safe(pd.DataFrame(advantage))
                st.dataframe(_style_status(df_adv, ["Your page"]), use_container_width=True)
                st.download_button(
                    "Download competitor advantage CSV", _csv_bytes(df_adv),
                    "competitor_advantage.csv", "text/csv",
                )
            elif matrix_rows:
                st.info("No entities found where competitors clearly outperform your page.")

        if failed:
            st.caption("Competitors that could not be analysed:")
            st.table(pd.DataFrame([{"URL": u, "Reason": r.get("error", "unknown")} for u, r in failed.items()]))

    if st.session_state.skipped:
        st.subheader("Skipped URLs")
        st.table(pd.DataFrame(st.session_state.skipped))


# =========================
# PHRASE FAN-OUT (standalone)
# =========================

st.divider()
st.header("Phrase fan-out")
st.markdown(
    "Generate fan-out queries for any list of phrases, no URL required. Useful for keyword "
    "research and content ideation. Uses the target country and max queries settings above."
)

phrases_input = st.text_area(
    "Enter one phrase per line:",
    placeholder="electric car charging\nsolar panel grants\nbest energy tariff",
    key="phrases_input",
)

if "phrase_results" not in st.session_state:
    st.session_state.phrase_results = []
if "phrase_processed" not in st.session_state:
    st.session_state.phrase_processed = False

if st.button("Generate fan-out queries", key="run_phrases_btn"):
    phrases_list = [p.strip() for p in phrases_input.splitlines() if p.strip()]
    if not phrases_list:
        st.warning("Please enter at least one phrase.")
    else:
        st.session_state.phrase_results = []
        st.session_state.phrase_processed = False
        progress = st.progress(0)
        status = st.empty()

        for i, phrase in enumerate(phrases_list, start=1):
            status.text(f"Processing {i}/{len(phrases_list)}: {phrase}")
            try:
                lvl1 = fanout.cached_fanouts(phrase, config.CANDIDATE_COUNT, config.GEMINI_TEMP, config.LVL1_TIMEOUT, country)
                lvl2 = fanout.expand_level2_parallel(
                    lvl1, attempts=config.ATTEMPTS, temp=config.GEMINI_TEMP,
                    max_workers=config.MAX_WORKERS, cand_count=config.LVL2_CANDIDATES,
                    timeout_s=config.LVL2_TIMEOUT, country=country,
                )
                all_qs = (lvl1 or []) + (lvl2 or [])
                if config.NORMALIZE_YEAR_SUFFIX:
                    all_qs = [text_utils.strip_trailing_years(q) for q in all_qs]
                all_qs = [text_utils.strip_geo_tokens(q) for q in all_qs]
                all_qs = [q for q in all_qs if q]

                if config.ENABLE_DEDUPE and all_qs:
                    reps, _g = text_utils.dedupe_pipeline(
                        all_qs, use_exact=True, fuzzy_ratio=config.FUZZY_RATIO,
                        use_embed=config.EMBED_ON, embed_threshold=config.EMBED_THRESHOLD,
                    )
                    final_qs = reps
                else:
                    seen, final_qs = set(), []
                    for q in all_qs:
                        if q.lower() not in seen:
                            seen.add(q.lower())
                            final_qs.append(q)

                for q in final_qs[:max_queries]:
                    st.session_state.phrase_results.append({"Phrase": phrase, "Fan-out query": q})
            except Exception as e:
                st.warning(f"Failed to generate fan-out for '{phrase}': {e}")

            progress.progress(i / len(phrases_list))

        status.text("Done.")
        st.session_state.phrase_processed = True

if st.session_state.phrase_processed and st.session_state.phrase_results:
    df_phrases = pd.DataFrame(st.session_state.phrase_results)
    st.subheader("Fan-out results")
    st.dataframe(df_phrases, use_container_width=True)
    st.download_button(
        "Download phrase fan-out CSV",
        df_phrases.to_csv(index=False).encode("utf-8"),
        "phrase_fanout.csv", "text/csv", key="download_phrase_csv",
    )

branding.render_footer()
