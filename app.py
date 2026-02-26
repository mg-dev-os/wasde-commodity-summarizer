"""Streamlit app: upload WASDE PDF, show commodity summaries."""

import tempfile
from pathlib import Path

import streamlit as st

from src.config_loader import get_config, get_gemini_api_key, get_groq_api_key, get_openai_api_key
from src.pdf_extractor import extract_text_by_commodity
from src.summarizer import summarize_all_commodities


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
    st.caption("Upload a USDA WASDE PDF to get AI summaries by commodity (Wheat, Coarse Grains, Rice, etc.).")

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
        with st.spinner("Extracting text from PDF..."):
            sections = extract_text_by_commodity(tmp_path)
        if not sections:
            st.warning("No commodity sections were found in this PDF. The file may be scanned or use a different structure.")
            return

        st.success(f"Found {len(sections)} section(s): {', '.join(sections.keys())}")

        if st.button("Generate summaries", type="primary"):
            with st.spinner("Summarizing..."):
                summaries = summarize_all_commodities(sections)

            st.divider()
            st.subheader("Summaries by commodity")
            for commodity, summary in summaries.items():
                with st.expander(commodity, expanded=True):
                    st.markdown(summary)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
