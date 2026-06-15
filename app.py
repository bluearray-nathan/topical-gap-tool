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
    "Map your page against the entities and queries AI search explores, across Google AI Overviews, "
    "AI Mode and ChatGPT, and against your competitors.",
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
# FAN-OUT HELPERS
# =========================

def _provider_list(engine_choice):
    if engine_choice.startswith("Both"):
        return [fanout.GEMINI, fanout.OPENAI]
    if engine_choice.startswith("ChatGPT"):
        return [fanout.OPENAI]
    return [fanout.GEMINI]


def _settings_row(prefix):
    """Render per-tab country / max queries / depth / engine controls. Returns
    (country, max_queries, providers, depth)."""
    c1, c2, c3, c4 = st.columns([1.2, 0.9, 0.9, 1.7])
    with c1:
        country = st.selectbox(
            "Target country:", config.COUNTRIES, index=0, key=f"{prefix}_country",
            help="Grounding searches and generated queries are focused on this country.",
        )
    with c2:
        max_queries = st.slider(
            "Max queries:", min_value=5, max_value=40, value=15, step=1, key=f"{prefix}_max",
            help="Upper limit on fan-out queries (after dedupe) per URL or topic.",
        )
    with c3:
        depth = st.slider(
            "Fan-out depth:", min_value=1, max_value=config.FANOUT_DEPTH_MAX,
            value=config.FANOUT_DEPTH_DEFAULT, step=1, key=f"{prefix}_depth",
            help="How many layers deep to fan out. 1 is direct queries only; each extra layer "
                 "expands the one before it. Deeper is broader but slower and uses more API calls, "
                 "and it pairs best with a higher max-queries cap.",
        )
    with c4:
        engine = st.radio(
            "Fan-out engine:",
            ["Gemini (grounded search)", "ChatGPT", "Both (combine)"],
            index=0, horizontal=True, key=f"{prefix}_engine",
            help="Gemini uses live Google Search grounding, which mirrors AI Mode. ChatGPT generates "
                 "its queries. Both runs each and merges them.",
        )
    return country, max_queries, _provider_list(engine), depth


def _fanout_reps(seed, providers, country, max_queries, depth):
    """Generate, dedupe and cap fan-out queries. Returns
    (raw, reps, groups, source_map, engine_counts)."""
    queries, source_map, engine_counts = fanout.generate_fanout(seed, country, providers, depth=depth)
    raw = queries[:]
    if config.ENABLE_DEDUPE and queries:
        reps, groups = text_utils.dedupe_pipeline(
            queries, use_exact=True, fuzzy_ratio=config.FUZZY_RATIO,
            use_embed=config.EMBED_ON, embed_threshold=config.EMBED_THRESHOLD,
        )
    else:
        seen, reps, groups = set(), [], {}
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                reps.append(q)
                groups[q] = [q]
    return raw, reps[:max_queries], groups, source_map, engine_counts


def _sources_for(rep, groups, source_map):
    s = set()
    for m in groups.get(rep, [rep]):
        s |= set(source_map.get(m.strip().lower(), []))
    return ", ".join(sorted(s))


# Tabs are the first thing on the page so neither feature is buried.
tab_audit, tab_keyword = st.tabs(["Page gap analysis", "Topic fan-out"])


# =========================
# TAB 1: PAGE GAP ANALYSIS
# =========================

with tab_audit:
    country, max_queries, providers, depth = _settings_row("audit")

    st.markdown(
        """
        <div class="ba-panel">
        <b>What this does</b><br>
        1. Reads the main content of each page (boilerplate, navigation and footers removed).<br>
        2. Generates multi-layer fan-out queries the way AI search explores a topic, using Gemini's grounded Google search, ChatGPT, or both.<br>
        3. Clusters those queries into <b>entities</b>: the named things and key subtopics a topic is built from.<br>
        4. Scores how well your page covers each entity (missing, thin or strong) with supporting evidence.<br>
        5. Optionally benchmarks your page against competitors to show which entities they cover and you do not.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    mode = st.radio(
        "Audit mode:",
        ["Single URL", "Multiple URLs"],
        horizontal=True, key="audit_mode",
        help="Single URL: deep-dive one page, with a target keyword and competitor benchmarking. "
             "Multiple URLs: audit a batch of pages at once, each seeded from its own H1.",
    )

    if mode == "Single URL":
        single_url = st.text_input("URL to audit:", placeholder="https://example.com/page")
        urls = [single_url.strip()] if single_url.strip() else []
        target_keyword = st.text_input(
            "Target keyword (optional):",
            placeholder="e.g. smart export guarantee",
            help="Seeds the fan-out and competitor discovery. If left blank, the page H1 is used.",
        )
        compare_competitors = st.checkbox("Benchmark against competitors", value=False)
        if compare_competitors:
            st.caption(
                "Benchmarks this page against competitors on the same entities. Each competitor "
                "adds a fetch and a scoring pass, so more competitors means a longer run."
            )
            if competitors.serpapi_available():
                if st.button("Suggest competitors from Google", key="suggest_comp"):
                    primary = urls[0] if urls else ""
                    seed = target_keyword.strip()
                    if not seed and primary:
                        with st.spinner("Reading the page to find a seed keyword..."):
                            seed = extraction.cached_extract_page(primary).h1
                    if not seed:
                        st.warning("Enter a target keyword or the URL first.")
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
    else:
        urls_input = st.text_area(
            "Enter one URL per line to audit:",
            placeholder="https://example.com/page1\nhttps://example.com/page2",
        )
        urls = [line.strip() for line in urls_input.splitlines() if line.strip()]
        target_keyword = ""
        compare_competitors = False
        st.caption(
            "Each page is seeded from its own H1. Target keyword and competitor benchmarking "
            "are available in Single URL mode."
        )

    # --- Session state ---
    def init_audit_state():
        defaults = {
            "last_inputs": None, "processed": False, "summary": [], "entity_maps": [],
            "actions": [], "skipped": [], "primary_url": "", "primary_entities": [],
            "competitor_results": {},
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    init_audit_state()

    current_inputs = (tuple(urls), target_keyword.strip(), country, max_queries, tuple(providers), mode, depth)
    if st.session_state.last_inputs != current_inputs:
        st.session_state.last_inputs = current_inputs
        st.session_state.processed = False
        for key in ("summary", "entity_maps", "actions", "skipped"):
            st.session_state[key].clear()
        st.session_state.primary_entities = []
        st.session_state.competitor_results = {}

    # --- Audit ---
    if st.button("Start audit") and urls and not st.session_state.processed:
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
            raw_queries, used_queries, _groups, _src, eng_counts = _fanout_reps(seed, providers, country, max_queries, depth)
            if not used_queries:
                st.warning(f"Skipped {url}: no fan-out queries generated")
                st.session_state.skipped.append({"Address": url, "Reason": "No queries generated"})
                progress.progress(int(((idx + 1) / total) * 100))
                continue

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
                "Address": url, "Words": page.word_count,
                "Gemini (raw)": eng_counts.get(fanout.GEMINI, 0),
                "ChatGPT (raw)": eng_counts.get(fanout.OPENAI, 0),
                "Queries used": len(used_queries), "Entities": len(rows),
                "Strong": counts["Strong"], "Thin": counts["Thin"], "Missing": counts["Missing"],
                "Entity coverage (%)": entity_mod.entity_coverage_percent(rows),
            })
            st.session_state.entity_maps.append({"url": url, "rows": rows})

            gaps = [f"{r['entity']} ({r['status']})" for r in rows if r["status"] != "Strong"]
            st.session_state.actions.append({
                "Address": url, "Entities to add or strengthen": "; ".join(gaps),
            })

            if idx == 0:
                st.session_state.primary_url = url
                st.session_state.primary_entities = ents

            progress.progress(int(((idx + 1) / total) * 100))

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

    # --- Results ---
    if st.session_state.processed:
        st.header("Results")

        if st.session_state.summary:
            st.subheader("Summary")
            df_summary = _safe(pd.DataFrame(st.session_state.summary))
            st.dataframe(df_summary, use_container_width=True)
            st.download_button("Download summary CSV", _csv_bytes(df_summary), "summary.csv", "text/csv")

        if st.session_state.entity_maps:
            st.subheader("Entity coverage map")
            st.caption("Sorted with missing and thin entities first. These are your content gaps.")
            combined = []
            for block in st.session_state.entity_maps:
                url = block["url"]
                display = pd.DataFrame([
                    {
                        "Entity": r["entity"], "Type": r["type"], "Status": r["status"],
                        "On schema": "Yes" if r["schema_supported"] else "",
                        "Evidence": r["evidence"], "Fan-out queries": "; ".join(r["queries"]),
                    }
                    for r in block["rows"]
                ])
                st.markdown(f"**{url}**")
                st.dataframe(_style_status(display, ["Status"]), use_container_width=True)
                for r in block["rows"]:
                    combined.append({
                        "Address": url, "Entity": r["entity"], "Type": r["type"], "Status": r["status"],
                        "On schema": "Yes" if r["schema_supported"] else "",
                        "Evidence": r["evidence"], "Fan-out queries": "; ".join(r["queries"]),
                    })
            df_entities = _safe(pd.DataFrame(combined))
            st.download_button("Download entity coverage CSV", _csv_bytes(df_entities), "entity_coverage.csv", "text/csv")

        if st.session_state.actions:
            st.subheader("Actions")
            df_actions = _safe(pd.DataFrame(st.session_state.actions))
            st.dataframe(df_actions, use_container_width=True)
            st.download_button("Download actions CSV", _csv_bytes(df_actions), "actions.csv", "text/csv")

        if st.session_state.competitor_results:
            st.subheader("Competitor benchmarking")
            primary_block = next(
                (b for b in st.session_state.entity_maps if b["url"] == st.session_state.primary_url), None
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
                    st.download_button("Download competitor matrix CSV", _csv_bytes(_safe(df_matrix)), "competitor_matrix.csv", "text/csv")
                advantage = competitors.competitor_advantage(primary_block["rows"], results)
                if advantage:
                    st.markdown("**Where competitors cover an entity and your page does not**")
                    df_adv = _safe(pd.DataFrame(advantage))
                    st.dataframe(_style_status(df_adv, ["Your page"]), use_container_width=True)
                    st.download_button("Download competitor advantage CSV", _csv_bytes(df_adv), "competitor_advantage.csv", "text/csv")
                elif matrix_rows:
                    st.info("No entities found where competitors clearly outperform your page.")
            if failed:
                st.caption("Competitors that could not be analysed:")
                st.table(pd.DataFrame([{"URL": u, "Reason": r.get("error", "unknown")} for u, r in failed.items()]))

        if st.session_state.skipped:
            st.subheader("Skipped URLs")
            st.table(pd.DataFrame(st.session_state.skipped))


# =========================
# TAB 2: KEYWORD FAN-OUT
# =========================

with tab_keyword:
    kw_country, kw_max, kw_providers, kw_depth = _settings_row("kw")

    st.markdown(
        "Enter a topic, such as solar panels, and this pulls in the relevant fan-out queries from "
        "Gemini, ChatGPT, or both, and groups them by entity. No URL required. Useful for keyword "
        "research and content planning."
    )

    phrases_input = st.text_area(
        "Enter one topic per line:",
        placeholder="solar panels\nelectric cars\nheat pump grants",
        key="phrases_input",
    )

    if "phrase_rows" not in st.session_state:
        st.session_state.phrase_rows = []
    if "phrase_engine_totals" not in st.session_state:
        st.session_state.phrase_engine_totals = {}

    if st.button("Generate topic fan-out", key="run_phrases_btn"):
        phrases = [p.strip() for p in phrases_input.splitlines() if p.strip()]
        if not phrases:
            st.warning("Please enter at least one topic.")
        else:
            st.session_state.phrase_rows = []
            totals = {fanout.GEMINI: 0, fanout.OPENAI: 0}
            progress = st.progress(0)
            status = st.empty()
            for i, phrase in enumerate(phrases, start=1):
                status.text(f"Processing {i}/{len(phrases)}: {phrase}")
                try:
                    _raw, reps, groups, source_map, eng_counts = _fanout_reps(phrase, kw_providers, kw_country, kw_max, kw_depth)
                    for _k in totals:
                        totals[_k] += eng_counts.get(_k, 0)
                    if not reps:
                        st.warning(f"No queries generated for '{phrase}'.")
                        continue
                    ents = entity_mod.cluster_entities(reps, topic=phrase, country=kw_country)
                    for e in ents:
                        for q in e["queries"]:
                            st.session_state.phrase_rows.append({
                                "Topic": phrase,
                                "Entity": e["name"],
                                "Type": e["type"],
                                "Fan-out query": q,
                                "Source": _sources_for(q, groups, source_map),
                            })
                except Exception as e:
                    st.warning(f"Failed to fan out '{phrase}': {e}")
                progress.progress(i / len(phrases))
            st.session_state.phrase_engine_totals = totals
            status.text("Done.")

    if st.session_state.phrase_rows:
        st.subheader("Topic fan-out grouped by entity")
        _tot = st.session_state.get("phrase_engine_totals", {})
        if _tot:
            st.caption(
                f"Generated before dedupe and cap: Gemini {_tot.get(fanout.GEMINI, 0)}, "
                f"ChatGPT {_tot.get(fanout.OPENAI, 0)}."
            )
        df_kw = pd.DataFrame(st.session_state.phrase_rows).sort_values(
            ["Topic", "Entity"]
        ).reset_index(drop=True)
        st.dataframe(df_kw, use_container_width=True)
        st.download_button(
            "Download topic fan-out CSV", _csv_bytes(_safe(df_kw)),
            "topic_fanout_entities.csv", "text/csv", key="dl_kw",
        )

branding.render_footer()
