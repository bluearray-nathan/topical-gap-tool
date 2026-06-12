"""Entity clustering and the entity coverage map.

Clusters the deduped fan-out queries into entities (a pragmatic mix of named
entities and key subtopics/attributes), then scores how well a page covers each
entity. Produces the headline deliverable: an entity coverage map.
"""

import json
import re

import openai
import streamlit as st

import config
import coverage

ENTITY_TYPES = [
    "brand", "product", "scheme", "regulation", "organisation", "place",
    "person", "cost", "eligibility", "process", "comparison", "feature", "other",
]

_SYSTEM = "You are an entity and topic analyst for SEO. You output ONLY a JSON array, no prose."


def _entity_bounds(n):
    """Sensible min/max entity counts for n queries."""
    lo = max(3, n // 4)
    hi = max(lo + 1, min(n, n // 2 + 1))
    return lo, hi


def _build_prompt(topic, queries, country):
    lo, hi = _entity_bounds(len(queries))
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(queries, start=1))
    return "\n".join([
        f"Topic: {topic}",
        f"Country context: {country}",
        "",
        "Below is a numbered list of search queries that AI search fans out to for this topic.",
        "Group them into ENTITIES: the key named things and the salient subtopics or attributes "
        "the topic is built from.",
        "",
        "Rules:",
        "- An entity is either a named entity (brand, product, scheme, regulation, organisation, "
        "place, person) or a key subtopic/attribute (cost, eligibility, process, comparison, feature).",
        "- Assign every query number to exactly one entity. Do not invent queries.",
        f"- Aim for {lo} to {hi} entities. Merge near-duplicates. Prefer section-level entities "
        "over bullet-level ones.",
        "- Give each entity a short name in sentence case, a type from this exact list "
        f"{ENTITY_TYPES}, and a one-line definition.",
        "",
        "Queries:",
        numbered,
        "",
        "Return ONLY a JSON array, e.g.",
        '[{"name":"...","type":"cost","definition":"...","query_numbers":[1,4]}]',
    ])


def _call_json(system, user, temperature=0.2, max_retries=2):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
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
    st.warning(f"Entity clustering parse failed. Raw response: {last[:300]}")
    return []


def cluster_entities(queries, topic, country="United Kingdom", temperature=0.2):
    """Group queries into entities. Returns list of
    {name, type, definition, queries:[...]}. Every query is preserved."""
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return []

    raw = _call_json(_SYSTEM, _build_prompt(topic, queries, country), temperature)
    entities, assigned = [], set()
    for node in raw:
        if not isinstance(node, dict):
            continue
        name = (node.get("name") or "").strip()
        if not name:
            continue
        etype = (node.get("type") or "other").strip().lower()
        if etype not in ENTITY_TYPES:
            etype = "other"
        nums = node.get("query_numbers") or node.get("queries") or []
        members = []
        for n in nums:
            try:
                idx = int(n) - 1
            except Exception:
                continue
            if 0 <= idx < len(queries):
                members.append(queries[idx])
                assigned.add(idx)
        if not members:
            continue
        entities.append({
            "name": name,
            "type": etype,
            "definition": (node.get("definition") or "").strip(),
            "queries": members,
        })

    # Preserve any queries the model failed to assign.
    leftover = [queries[i] for i in range(len(queries)) if i not in assigned]
    if leftover:
        entities.append({
            "name": "Other queries",
            "type": "other",
            "definition": "Queries not grouped into a named entity.",
            "queries": leftover,
        })
    return entities


def _schema_supported(entity_name, page):
    name = entity_name.lower()
    tokens = [t for t in re.findall(r"\w+", name) if len(t) > 3]
    haystack = " ".join(
        [e["name"].lower() for e in page.jsonld_entities]
        + [txt.lower() for _lvl, txt in page.headings]
    )
    if not haystack:
        return False
    if name in haystack:
        return True
    return bool(tokens) and all(t in haystack for t in tokens)


_STATUS_ORDER = {"Missing": 0, "Thin": 1, "Strong": 2}


def build_entity_map(page, entities):
    """Score each entity's coverage on the page and return a sorted map
    (missing and thin first)."""
    if not entities:
        return []
    items = [
        {"name": e["name"], "queries": e["queries"], "definition": e["definition"]}
        for e in entities
    ]
    scored = coverage.score_coverage(page, items)
    by_label = {r["label"].strip().lower(): r for r in scored}

    rows = []
    for e in entities:
        r = by_label.get(e["name"].strip().lower(), {"score": 0, "status": "Missing", "evidence": ""})
        rows.append({
            "entity": e["name"],
            "type": e["type"],
            "definition": e["definition"],
            "score": r["score"],
            "status": r["status"],
            "evidence": r["evidence"],
            "queries": e["queries"],
            "schema_supported": _schema_supported(e["name"], page),
        })
    rows.sort(key=lambda x: (_STATUS_ORDER.get(x["status"], 0), x["entity"].lower()))
    return rows


def entity_coverage_percent(rows):
    if not rows:
        return 0
    return int(round(100 * sum(r["score"] for r in rows) / (2 * len(rows))))
