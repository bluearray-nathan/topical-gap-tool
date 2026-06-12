"""Query normalisation, dedupe and embedding helpers.

Moved from the original single-file app unchanged in behaviour, so the dedupe
pipeline and normalisation produce identical results.
"""

import re
from datetime import datetime

import numpy as np
import openai
import streamlit as st

import config

# Optional fuzzy matching (RapidFuzz). Falls back to a simple Jaccard if unavailable.
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False


# =========================
# NORMALISATION
# =========================

# Handles endings like: " ... 2024", " ... (2024)", " ... in 2024", " ... 2024/25", " ... 2024-2025"
YEAR_SUFFIX_RE = re.compile(
    r"""
    (?:\s*[\(\-]?\s*)        # optional spacing / '(' / '-' before the year
    (?:in\s+)?               # optional 'in '
    ((?:19|20)\d{2})         # base 4-digit year (capture)
    (?:\s*/\s*\d{2}          # ' /25' short range
       |-(?:19|20)\d{2}      # or '-2025' full range
    )?                       # optional year-range
    \)?\s*$                  # optional ')' then end of string
    """,
    re.IGNORECASE | re.VERBOSE,
)


def strip_trailing_years(q: str, min_year: int = 2000, max_year: int | None = None) -> str:
    """Remove a trailing year or year-range only at the end of the query."""
    if max_year is None:
        max_year = datetime.now().year + 1  # allow next-year planning queries
    m = YEAR_SUFFIX_RE.search(q)
    if not m:
        return q
    try:
        y = int(m.group(1))
    except Exception:
        return q
    if min_year <= y <= max_year:
        return q[:m.start()].rstrip()
    return q


# Standalone country/region tokens that should be stripped from queries.
GEO_TOKENS_RE = re.compile(
    r"\b(uk|u\.k\.|gb|gbr|great britain|britain|united kingdom|"
    r"us|u\.s\.|usa|u\.s\.a\.|united states|america|"
    r"ca|can|canada|"
    r"au|aus|australia|"
    r"ie|ireland|"
    r"nz|new zealand|"
    r"de|germany|deutschland|"
    r"fr|france|"
    r"es|spain|espana|españa|"
    r"it|italy|italia|"
    r"nl|netherlands|holland|"
    r"za|south africa|"
    r"in|india|"
    r"uae|united arab emirates|"
    r"sg|singapore)\b",
    re.IGNORECASE,
)


def strip_geo_tokens(q: str) -> str:
    """Remove standalone country/region tokens from a query and tidy whitespace."""
    cleaned = GEO_TOKENS_RE.sub("", q)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -,")
    return cleaned


# =========================
# DEDUPE
# =========================

STOPWORDS = {
    "the", "and", "of", "in", "to", "a", "for", "with", "on", "about",
    "vs", "vs.", "is", "are", "your", "what", "how", "why", "more",
    "latest", "new", "does", "do", "an",
}


def canonicalize(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^\w\s]", " ", q)
    if STOPWORDS:
        q = re.sub(r"\b(" + "|".join(map(re.escape, STOPWORDS)) + r")\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _token_set_ratio(a: str, b: str) -> int:
    if HAVE_RAPIDFUZZ:
        return int(fuzz.token_set_ratio(a, b))
    ta, tb = set(canonicalize(a).split()), set(canonicalize(b).split())
    if not ta and not tb:
        return 100
    if not ta or not tb:
        return 0
    jacc = len(ta & tb) / len(ta | tb)
    return int(round(jacc * 100))


def dedupe_exact(queries):
    seen = set()
    kept, groups = [], {}
    for q in queries:
        c = canonicalize(q)
        if c in seen:
            for rep in kept:
                if canonicalize(rep) == c:
                    groups[rep].append(q)
                    break
        else:
            seen.add(c)
            kept.append(q)
            groups[q] = [q]
    return kept, groups


def dedupe_token_set(queries, min_ratio=92):
    kept = []
    groups = {}
    for q in queries:
        attached = False
        for k in kept:
            score = _token_set_ratio(q, k)
            if score >= min_ratio:
                groups[k].append(q)
                attached = True
                break
        if not attached:
            kept.append(q)
            groups[q] = [q]
    return kept, groups


def get_embeddings(queries, model=None):
    model = model or config.embed_model()
    resp = openai.embeddings.create(model=model, input=queries)
    data = getattr(resp, "data", None) or resp["data"]
    out = []
    for d in data:
        emb = getattr(d, "embedding", None)
        if emb is None:
            emb = d.get("embedding", [])
        out.append(np.array(emb, dtype=float))
    return out


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def dedupe_embeddings(queries, threshold=0.86, model=None):
    if not queries:
        return [], {}
    try:
        vecs = get_embeddings(queries, model=model)
    except Exception as e:
        st.warning(f"Embedding dedupe skipped (embedding error): {e}")
        return queries, {q: [q] for q in queries}
    kept, groups, removed = [], {}, set()
    for i, qi in enumerate(queries):
        if i in removed:
            continue
        kept.append(qi)
        groups[qi] = [qi]
        for j in range(i + 1, len(queries)):
            if j in removed:
                continue
            sim = cosine(vecs[i], vecs[j])
            if sim >= threshold:
                groups[qi].append(queries[j])
                removed.add(j)
    return kept, groups


def dedupe_pipeline(
    queries,
    use_exact=True,
    fuzzy_ratio=92,
    use_embed=True,
    embed_threshold=0.86,
    embed_model=None,
):
    if not queries:
        return [], {}
    q = queries[:]
    if use_exact:
        q, groups_exact = dedupe_exact(q)
    else:
        groups_exact = {x: [x] for x in q}
    q2, groups_ts = dedupe_token_set(q, min_ratio=fuzzy_ratio)
    merged_ts = {rep: [] for rep in q2}
    for rep in q2:
        merged_ts[rep].extend(groups_ts[rep])
    if use_embed and len(q2) > 1:
        reps, sem_groups = dedupe_embeddings(q2, threshold=embed_threshold, model=embed_model)
        final_groups = {r: [] for r in reps}
        for r in reps:
            for member in sem_groups[r]:
                final_groups[r].extend(merged_ts.get(member, [member]))
        for r in final_groups:
            seen, ded = set(), []
            for item in final_groups[r]:
                if item not in seen:
                    seen.add(item)
                    ded.append(item)
            final_groups[r] = ded
        return reps, final_groups
    return q2, merged_ts
