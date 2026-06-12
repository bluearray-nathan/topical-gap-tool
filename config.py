"""Central configuration: secrets, model names, proxy settings and tunable constants.

Secret and model lookups are lazy and exception-safe, so these helpers can be
imported (and unit-tested) outside a running Streamlit session without raising.
"""

import os
import openai
import streamlit as st

DEFAULT_USER_AGENT = "QueryFanOutBot/1.0"


# =========================
# SECRET / ENV LOOKUP
# =========================

def _secret_value(path, default=None):
    """Safely read a nested value from st.secrets, falling back to ``default``.

    Never raises, even when no secrets file is present.
    """
    try:
        current = st.secrets
        for key in path:
            if key not in current:
                return default
            current = current[key]
        return current
    except Exception:
        return default


def get_openai_key():
    return _secret_value(("openai", "api_key")) or os.getenv("OPENAI_API_KEY")


def get_gemini_key():
    return _secret_value(("google", "gemini_api_key")) or os.getenv("GEMINI_API_KEY")


def get_serpapi_key():
    """Optional. When absent, competitor auto-discovery is disabled gracefully."""
    return _secret_value(("serpapi", "api_key")) or os.getenv("SERPAPI_API_KEY")


# =========================
# MODEL NAMES (override-able)
# =========================
# Override via a [models] block in secrets, or env vars, so a model rename
# never silently breaks the app.

def gemini_model():
    return (
        _secret_value(("models", "gemini"))
        or os.getenv("GEMINI_MODEL")
        or "gemini-3-flash-preview"
    )


def openai_model():
    return (
        _secret_value(("models", "openai"))
        or os.getenv("OPENAI_MODEL")
        or "gpt-5.4-mini"
    )


def embed_model():
    return (
        _secret_value(("models", "embedding"))
        or os.getenv("EMBED_MODEL")
        or "text-embedding-3-small"
    )


def configure_apis():
    """Set the OpenAI key on the module-level client. Call once at app start."""
    key = get_openai_key()
    if key:
        openai.api_key = key


# =========================
# PROXY / USER AGENT
# =========================

def get_proxy_url():
    return (
        _secret_value(("proxy", "server"))
        or os.getenv("PROXY_SERVER")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )


def get_proxy_username():
    return _secret_value(("proxy", "username")) or os.getenv("PROXY_USERNAME")


def get_proxy_password():
    return _secret_value(("proxy", "password")) or os.getenv("PROXY_PASSWORD")


def get_user_agent():
    return (
        _secret_value(("crawler", "user_agent"))
        or os.getenv("CRAWLER_USER_AGENT")
        or DEFAULT_USER_AGENT
    )


def get_requests_proxy_map():
    proxy_url = get_proxy_url()
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


def get_playwright_proxy_settings():
    proxy_url = get_proxy_url()
    if not proxy_url:
        return None
    proxy = {"server": proxy_url}
    username = get_proxy_username()
    password = get_proxy_password()
    if username:
        proxy["username"] = username
    if password:
        proxy["password"] = password
    return proxy


# =========================
# TUNABLE CONSTANTS
# =========================

# Fan-out generation
GEMINI_TEMP = 0.4
ATTEMPTS = 1
CANDIDATE_COUNT = 7
LVL1_TIMEOUT = 30
LVL2_CANDIDATES = 4
LVL2_TIMEOUT = 15
MAX_WORKERS = 6

# Coverage scoring
GPT_TEMP = 0.1
COVERAGE_BATCH_SIZE = 8
# Pages longer than this (characters of clean main content) switch to
# embedding retrieval per item rather than sending the whole body.
BODY_FULL_LIMIT = 14000
RETRIEVAL_CHUNK_CHARS = 1200
RETRIEVAL_TOP_K = 6

# Dedupe
ENABLE_DEDUPE = True
FUZZY_RATIO = 92
EMBED_ON = True
EMBED_THRESHOLD = 0.86

# Normalisation
NORMALIZE_YEAR_SUFFIX = True

# Competitors
MAX_COMPETITORS = 5
COMPETITOR_TIMEOUT = 20

COUNTRIES = [
    "United Kingdom",
    "United States",
    "Canada",
    "Australia",
    "Ireland",
    "New Zealand",
    "Germany",
    "France",
    "Spain",
    "Italy",
    "Netherlands",
    "South Africa",
    "India",
    "United Arab Emirates",
    "Singapore",
]

# SerpAPI country codes (gl parameter) for the supported countries.
SERP_COUNTRY_CODE = {
    "United Kingdom": "uk",
    "United States": "us",
    "Canada": "ca",
    "Australia": "au",
    "Ireland": "ie",
    "New Zealand": "nz",
    "Germany": "de",
    "France": "fr",
    "Spain": "es",
    "Italy": "it",
    "Netherlands": "nl",
    "South Africa": "za",
    "India": "in",
    "United Arab Emirates": "ae",
    "Singapore": "sg",
}
