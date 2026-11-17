"""Dynamic Concept Explorer: Streamlit multipage entrypoint."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Dynamic Concept Explorer",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Shared session state defaults across pages
if "wiki_url" not in st.session_state:
    st.session_state["wiki_url"] = ""
if "pending_explore" not in st.session_state:
    st.session_state["pending_explore"] = False
if "exploration" not in st.session_state:
    st.session_state["exploration"] = None

page = st.navigation(
    [
        st.Page(
            "app_pages/home.py",
            title="Home",
            icon=":material/home:",
            default=True,
            help="Enter a full Wikipedia article URL to launch the concept network explorer.",
        ),
        st.Page(
            "app_pages/explorer.py",
            title="Explorer",
            icon=":material/hub:",
            help=(
                "Interactive concept graph, article inspector, visualizations, "
                "summarizer benchmark, and quantitative evaluation."
            ),
        ),
    ],
    position="top",
)

page.run()
