"""Graded coverage scoring.

Replaces the original binary covered/not-covered judgement with a 0-2 score
(missing / thin / strong) plus a short supporting quote from the page. The quote
requirement curbs the model claiming coverage that isn't in the text.

Works on the full main content. For long pages it retrieves the most relevant
extracts per item via embeddings, so page length stops causing false gaps.
"""

import json
import re

import numpy as np
import openai
import streamlit as st

import config
import text_utils

STATUS = {0: "Missing", 1: "Thin", 2: "Strong"}

_SYSTEM = "You are an SEO content gap auditor. You output ONLY a JSON array, no prose."


def _normalise_items(items):
    """Accept strings or dicts; return uniform {label, look_for, retrieval_text}."""
    norm = []
    for it in items:
        if isinstance(it, str):
            norm.append({"label": it, "look_for": it, "retrieval_text": it})
            continue
        label = it.get("label") or it.get("name") or ""
        look = it.get("look_for")
        if not look:
            parts = []
            if it.get("definition"):
                parts.append(str(it["definition"]))
            qs = it.get("queries") or []
            if qs:
                parts.append("Addresses these queries: " + "; ".join(qs))
            look = ". ".join(parts) or label
        rt = it.get("retrieval_text") or (label + " " + " ".join(it.get("queries") or [])).strip()
        norm.append({"label": label, "look_for": look, "retrieval_text": rt})
    return norm


def _chunk_text(text, chunk_chars):
    chunks, buf = [], ""
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 1 > chunk_chars and buf:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def _page_header(page):
    lines = [f"Page topic (H1): {page.h1}"]
    if page.headings:
        lines.append("Headings:")
        lines += [f"- {lvl}: {txt}" for lvl, txt in page.headings]
    if page.faqs:
        lines.append("Questions answered on page:")
        lines += [f"- {q}" for q in page.faqs[:25]]
    return "\n".join(lines)


def _relevant_body(page, norm_items):
    """Full body for short pages; retrieved extracts for long ones."""
    body = page.body or ""
    if len(body) <= config.BODY_FULL_LIMIT:
        return body
    chunks = _chunk_text(body, config.RETRIEVAL_CHUNK_CHARS)
    if len(chunks) <= config.RETRIEVAL_TOP_K:
        return body
    try:
        chunk_vecs = text_utils.get_embeddings(chunks)
        item_vecs = text_utils.get_embeddings([it["retrieval_text"] for it in norm_items])
    except Exception:
        return body[: config.BODY_FULL_LIMIT]
    selected = set()
    for iv in item_vecs:
        sims = [text_utils.cosine(iv, cv) for cv in chunk_vecs]
        top = np.argsort(sims)[::-1][: config.RETRIEVAL_TOP_K]
        selected.update(int(i) for i in top)
    ordered = [chunks[i] for i in sorted(selected)]
    return "\n\n".join(ordered)


def _build_prompt(page, body_text, batch):
    lines = [
        "Assess how well the PAGE below covers each ITEM.",
        "",
        "For each item return an object with:",
        '- "label": the item label copied exactly',
        '- "score": 0 if the page does not cover it, 1 if mentioned only briefly or partially, '
        "2 if covered well with substantive detail",
        '- "evidence": a short verbatim quote (max ~20 words) from the page supporting a score of '
        "1 or 2, or an empty string when the score is 0",
        "",
        "Be strict. Only give 1 or 2 when the content is actually present in the text provided. "
        "Do not infer coverage from the topic alone.",
        "",
        "PAGE CONTENT",
        "============",
        _page_header(page),
        "",
        "Main content:",
        body_text,
        "",
        "ITEMS",
        "=====",
    ]
    for it in batch:
        lines.append(f"- label: {it['label']}")
        if it["look_for"] and it["look_for"] != it["label"]:
            lines.append(f"  what to look for: {it['look_for']}")
    lines += [
        "",
        "Return ONLY a JSON array, e.g.",
        '[{"label":"...","score":2,"evidence":"..."}]',
    ]
    return "\n".join(lines)


def _call_llm(prompt, temperature, max_retries=2):
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": prompt},
    ]
    last = ""
    for _ in range(max_retries):
        try:
            resp = openai.chat.completions.create(
                model=config.openai_model(), messages=messages, temperature=temperature
            )
            text = (resp.choices[0].message.content or "").strip()
            last = text
            match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
    st.warning(f"Coverage parse failed. Raw response: {last[:300]}")
    return []


def _coerce_score(val):
    try:
        s = int(val)
    except Exception:
        s = 0
    return max(0, min(2, s))


def _align(norm_items, raw_results):
    """Map LLM rows back to input items by label, then by position."""
    by_label = {}
    for r in raw_results:
        if isinstance(r, dict) and r.get("label"):
            by_label[str(r["label"]).strip().lower()] = r
    out = []
    for idx, it in enumerate(norm_items):
        r = by_label.get(it["label"].strip().lower())
        if r is None and idx < len(raw_results) and isinstance(raw_results[idx], dict):
            r = raw_results[idx]
        score = _coerce_score(r.get("score")) if r else 0
        evidence = (r.get("evidence") if r else "") or ""
        out.append({
            "label": it["label"],
            "score": score,
            "status": STATUS[score],
            "evidence": str(evidence).strip(),
        })
    return out


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def score_coverage(page, items, batch_size=None, temperature=None):
    """Score how well ``page`` covers each item. Returns list of
    {label, score (0-2), status, evidence}."""
    if not items:
        return []
    batch_size = batch_size or config.COVERAGE_BATCH_SIZE
    temperature = config.GPT_TEMP if temperature is None else temperature
    norm = _normalise_items(items)
    results = []
    for batch in _chunked(norm, batch_size):
        body_text = _relevant_body(page, batch)
        prompt = _build_prompt(page, body_text, batch)
        raw = _call_llm(prompt, temperature)
        results.extend(_align(batch, raw))
    return results


def coverage_percent(results):
    """Weighted coverage: full credit for strong, half for thin."""
    if not results:
        return 0
    return int(round(100 * sum(r["score"] for r in results) / (2 * len(results))))


def status_counts(results):
    counts = {"Strong": 0, "Thin": 0, "Missing": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts
