"""Blue Array brand styling for the Streamlit app (2026 palette).

Fonts: Source Serif 4 (semi-bold) for headings, Raleway (regular) for body.
Palette and rules follow the Blue Array brand guidelines. No decorative shapes,
sentence case headings, British English copy, no grey body text.
"""

import base64
import os

import streamlit as st

# 2026 palette
PRIMARY_BLUE = "#1291D2"
DARK_NAVY = "#002140"
CORAL = "#EC4E64"
LIGHT_BLUE_BG = "#EAF1F9"
LIGHT_GREY = "#F2F2F2"
WHITE = "#FFFFFF"

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _logo_data_uri(filename):
    path = os.path.join(_ASSETS, filename)
    try:
        with open(path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def inject_css():
    """Load brand fonts and restyle Streamlit components. Call once at app start."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Raleway:wght@400;500;600&display=swap');

        html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
        .stApp, .stMarkdown, p, li, label, span, div, input, textarea, select, button {{
            font-family: 'Raleway', sans-serif;
        }}

        h1, h2, h3, h4, h5, h6,
        [data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {{
            font-family: 'Source Serif 4', Georgia, serif !important;
            font-weight: 600 !important;
            color: {DARK_NAVY};
            letter-spacing: -0.01em;
        }}

        .stApp {{ color: {DARK_NAVY}; }}

        a, a:visited {{ color: {PRIMARY_BLUE}; }}

        /* Primary action buttons */
        div.stButton > button, div.stDownloadButton > button {{
            background-color: {PRIMARY_BLUE};
            color: {WHITE};
            border: none;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
            transition: background-color 0.15s ease-in-out;
        }}
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            background-color: {DARK_NAVY};
            color: {WHITE};
        }}
        div.stButton > button:focus, div.stDownloadButton > button:focus {{
            box-shadow: 0 0 0 2px rgba(18, 145, 210, 0.4);
            color: {WHITE};
        }}

        /* Slider + selection accents */
        [data-testid="stSlider"] [role="slider"] {{ background-color: {PRIMARY_BLUE}; }}
        [data-baseweb="tag"] {{ background-color: {PRIMARY_BLUE} !important; }}

        /* Tabs */
        [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: {PRIMARY_BLUE};
            border-bottom-color: {PRIMARY_BLUE};
        }}

        /* Metric values */
        [data-testid="stMetricValue"] {{ color: {DARK_NAVY}; font-family: 'Source Serif 4', serif; }}

        /* Brand header panel */
        .ba-header {{
            background-color: {DARK_NAVY};
            border-radius: 10px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 1.4rem;
        }}
        .ba-header img {{ height: 46px; width: auto; }}
        .ba-header .ba-divider {{
            width: 1px; height: 46px; background: rgba(255,255,255,0.25);
        }}
        .ba-header .ba-title {{
            font-family: 'Source Serif 4', serif;
            font-weight: 600;
            font-size: 1.55rem;
            color: {WHITE};
            line-height: 1.2;
        }}
        .ba-header .ba-subtitle {{
            font-family: 'Raleway', sans-serif;
            font-weight: 400;
            font-size: 0.95rem;
            color: {WHITE};
            opacity: 0.85;
            margin-top: 0.15rem;
        }}

        /* Soft info panel reused across the app */
        .ba-panel {{
            background-color: {LIGHT_BLUE_BG};
            border-radius: 8px;
            padding: 1rem 1.2rem;
        }}
        .ba-footer {{
            color: {DARK_NAVY};
            opacity: 0.7;
            font-size: 0.85rem;
            margin-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title, subtitle=None):
    """Navy header band with the reversed Blue Array logo and the tool title."""
    logo = _logo_data_uri("blue-array-logo-white.png")
    logo_html = f'<img src="{logo}" alt="Blue Array" />' if logo else ""
    divider = '<div class="ba-divider"></div>' if logo else ""
    sub_html = f'<div class="ba-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="ba-header">
            {logo_html}
            {divider}
            <div>
                <div class="ba-title">{title}</div>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        '<div class="ba-footer">Blue Array &middot; Search that converts.</div>',
        unsafe_allow_html=True,
    )
