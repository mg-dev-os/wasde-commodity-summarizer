"""Streamlit app: upload WASDE PDF, show commodity summaries."""

import tempfile
from pathlib import Path

import streamlit as st

from src.config_loader import get_config, get_gemini_api_key, get_groq_api_key, get_openai_api_key
from src.pdf_extractor import extract_text_by_commodity, extract_raw_text
from src.summarizer import summarize_all_commodities, answer_query


def main():
    config = get_config()
    ui = config.get("ui", {})
    title = ui.get("page_title", "WASDE Commodity Summarizer")
    max_mb = ui.get("max_file_size_mb", 50)
    max_bytes = max_mb * 1024 * 1024

    st.set_page_config(page_title=title, layout="wide", menu_items={})
    # Hide Streamlit header toolbar (Deploy + three-dot menu)
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            header [data-testid="stToolbar"] {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(title)
    st.caption("Upload a USDA WASDE PDF to get AI summaries by commodity (Wheat, Coarse Grains, Rice, etc.) and search the document with natural language questions.")

    config = get_config()
    provider = (config.get("llm") or {}).get("provider", "openai")
    if provider == "openai" and not get_openai_api_key():
        st.error("OpenAI is selected but OPENAI_API_KEY is not set. Add it to `.env`, or set `llm.provider` to `groq` or `gemini` in config.yaml for free cloud summaries.")
        return
    if provider == "groq" and not get_groq_api_key():
        st.error("Groq is selected but GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys and add it to `.env`.")
        return
    if provider == "gemini" and not get_gemini_api_key():
        st.error("Gemini is selected but GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/app/apikey and add it to `.env`.")
        return
    if provider == "groq":
        st.info("Using **Groq** (free cloud). Summaries will be generated via Groq API.")
    elif provider == "gemini":
        st.info("Using **Google Gemini** (free tier). Summaries will be generated via Gemini API.")

    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], help="WASDE reports from usda.gov/oce/commodity/wasde")
    if not uploaded:
        st.info("Upload a PDF to begin.")
        return

    if uploaded.size > max_bytes:
        st.error(f"File too large. Maximum size: {max_mb} MB.")
        return

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    try:
        # Cache extracted text per file so we don't re-extract on every Search/Summarize click
        cache_key = (uploaded.name, uploaded.size)
        if (
            st.session_state.get("pdf_cache_key") == cache_key
            and st.session_state.get("cached_sections") is not None
            and st.session_state.get("cached_full_text") is not None
        ):
            sections = st.session_state["cached_sections"]
            full_text = st.session_state["cached_full_text"]
        else:
            with st.spinner("Extracting text from PDF..."):
                sections = extract_text_by_commodity(tmp_path)
                full_text = extract_raw_text(tmp_path)
            st.session_state["pdf_cache_key"] = cache_key
            st.session_state["cached_sections"] = sections
            st.session_state["cached_full_text"] = full_text

        if not sections:
            st.warning("No commodity sections were found in this PDF. The file may be scanned or use a different structure.")
            return

        st.success(f"Found {len(sections)} section(s): {', '.join(sections.keys())}")

        # When user uploads a different file, clear cache and previous summaries/search result
        if "uploaded_file_name" not in st.session_state:
            st.session_state["uploaded_file_name"] = None
        if st.session_state["uploaded_file_name"] != uploaded.name:
            st.session_state["uploaded_file_name"] = uploaded.name
            st.session_state["summaries"] = None
            st.session_state["search_answer"] = None
        if "summaries" not in st.session_state:
            st.session_state["summaries"] = None
        if "search_answer" not in st.session_state:
            st.session_state["search_answer"] = None

        # Two columns so Summaries and Search are visible at the same time
        col_summaries, col_search = st.columns([1, 1])

        with col_summaries:
            st.subheader("Summaries by commodity")
            if st.button("Generate summaries", type="primary", key="btn_summaries"):
                with st.spinner("Summarizing..."):
                    st.session_state["summaries"] = summarize_all_commodities(sections)
            if st.session_state["summaries"]:
                for commodity, summary in st.session_state["summaries"].items():
                    with st.expander(commodity, expanded=True):
                        st.markdown(summary)
            else:
                st.info("Click **Generate summaries** to see summaries for each commodity.")

        with col_search:
            st.subheader("Search the document")
            st.caption("Ask a natural language question about the uploaded PDF.")
            search_query = st.text_input("Your question", key="search_query", placeholder="e.g. What are the wheat export projections?")
            if st.button("Search", key="btn_search"):
                if search_query and search_query.strip():
                    with st.spinner("Searching..."):
                        st.session_state["search_answer"] = answer_query(search_query, full_text)
                else:
                    st.session_state["search_answer"] = None
                    st.info("Enter a question and click Search.")
            if st.session_state.get("search_answer") is not None:
                st.markdown("**Answer:**")
                st.markdown(st.session_state["search_answer"])
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
