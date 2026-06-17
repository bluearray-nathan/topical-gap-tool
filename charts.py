"""Brand-coloured Altair charts for the app's results views.

Altair ships with Streamlit, so this needs no extra dependency. Charts use the
2026 palette only and the Raleway body font, and each returns None when there is
nothing to plot so the caller can skip it cleanly.
"""

import re
from collections import Counter

import altair as alt
import pandas as pd

PRIMARY_BLUE = "#1291D2"
LIGHT_BLUE = "#8FBEE0"   # tint of primary, used for the "thin" / mid band
CORAL = "#EC4E64"
DARK_NAVY = "#002140"
BODY_FONT = "Raleway"

STATUS_DOMAIN = ["Strong", "Thin", "Missing"]
STATUS_RANGE = [PRIMARY_BLUE, LIGHT_BLUE, CORAL]
BAND_DOMAIN = ["Below 60%", "60 to 79%", "80% and above"]
BAND_RANGE = [CORAL, LIGHT_BLUE, PRIMARY_BLUE]


def _slug(address):
    s = str(address).rstrip("/")
    s = s.split("/article/")[-1] if "/article/" in s else s.split("/")[-1]
    s = s.replace("-", " ").replace("_", " ").strip()
    return (s[:1].upper() + s[1:])[:48] if s else str(address)


def _band(cov):
    try:
        cov = float(cov)
    except Exception:
        return "60 to 79%"
    if cov < 60:
        return "Below 60%"
    if cov < 80:
        return "60 to 79%"
    return "80% and above"


def _brand(chart):
    """Apply the body font across axes, legend and tooltip text."""
    return (
        chart.configure_axis(labelFont=BODY_FONT, titleFont=BODY_FONT, labelColor=DARK_NAVY, titleColor=DARK_NAVY)
        .configure_legend(labelFont=BODY_FONT, titleFont=BODY_FONT)
        .configure_view(strokeWidth=0)
    )


def coverage_bar(summary_rows):
    """Pages ranked by entity coverage %, colour-banded."""
    rows = []
    for r in summary_rows:
        cov = r.get("Entity coverage (%)")
        rows.append({"Page": _slug(r.get("Address", "")), "Coverage": cov, "Band": _band(cov)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Coverage:Q", title="Entity coverage (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Page:N", sort="x", title=None),
            color=alt.Color("Band:N", title="Coverage band",
                            scale=alt.Scale(domain=BAND_DOMAIN, range=BAND_RANGE)),
            tooltip=["Page", "Coverage"],
        )
        .properties(height=max(200, 22 * len(df)))
    )
    return _brand(chart)


def status_donut(strong, thin, missing):
    """Entity status mix across all audited pages."""
    if (strong + thin + missing) == 0:
        return None
    df = pd.DataFrame([
        {"Status": "Strong", "Count": strong},
        {"Status": "Thin", "Count": thin},
        {"Status": "Missing", "Count": missing},
    ])
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Status:N", title="Status",
                            scale=alt.Scale(domain=STATUS_DOMAIN, range=STATUS_RANGE)),
            tooltip=["Status", "Count"],
        )
        .properties(height=260)
    )
    return _brand(chart)


def gaps_by_type(entity_rows):
    """Count of missing or thin entities by entity type."""
    counts = Counter()
    for r in entity_rows:
        if r.get("status") in ("Missing", "Thin"):
            counts[r.get("type", "other")] += 1
    if not counts:
        return None
    df = pd.DataFrame([{"Type": t, "Gaps": n} for t, n in counts.items()])
    chart = (
        alt.Chart(df)
        .mark_bar(color=PRIMARY_BLUE)
        .encode(
            x=alt.X("Gaps:Q", title="Missing or thin entities"),
            y=alt.Y("Type:N", sort="-x", title=None),
            tooltip=["Type", "Gaps"],
        )
        .properties(height=max(180, 26 * len(df)))
    )
    return _brand(chart)


def competitor_status_bars(primary_rows, competitor_results, domain_of, primary_url):
    """Stacked entity status per site: your page versus each competitor."""
    rows = []
    yc = Counter(r.get("status") for r in primary_rows)
    for s in STATUS_DOMAIN:
        rows.append({"Site": "Your page", "Status": s, "Count": yc.get(s, 0)})
    for url, res in competitor_results.items():
        if not res.get("ok"):
            continue
        cc = Counter(cell.get("status") for cell in res.get("scores", {}).values())
        for s in STATUS_DOMAIN:
            rows.append({"Site": domain_of(url), "Status": s, "Count": cc.get(s, 0)})
    if len(rows) <= len(STATUS_DOMAIN):  # only your page, no competitors
        return None
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Site:N", title=None, sort=None),
            y=alt.Y("Count:Q", title="Entities", stack=True),
            color=alt.Color("Status:N", title="Status",
                            scale=alt.Scale(domain=STATUS_DOMAIN, range=STATUS_RANGE)),
            order=alt.Order("Status:N"),
            tooltip=["Site", "Status", "Count"],
        )
        .properties(height=320)
    )
    return _brand(chart)
