"""Page 1: Minimal landing page with header and Wikipedia URL input."""

from __future__ import annotations

import streamlit as st

from dynamic_concept_explorer.explorer_helpers import validate_wikipedia_url

st.title("Dynamic Concept Explorer")

with st.form("home_url_form"):
    wiki_url = st.text_input(
        "Wikipedia article URL",
        value=st.session_state.get("wiki_url", ""),
        placeholder="https://en.wikipedia.org/wiki/Machine_learning",
        key="home_wiki_url_input",
    )
    submitted = st.form_submit_button(
        "Explore",
        type="primary",
        icon=":material/search:",
        width="content",
    )

if submitted:
    try:
        canonical_url, _ = validate_wikipedia_url(wiki_url)
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:")
    else:
        st.session_state["wiki_url"] = canonical_url
        st.session_state["pending_explore"] = True
        st.session_state["exploration"] = None
        st.switch_page("app_pages/explorer.py")
