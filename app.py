import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from backend import analyze_resume

load_dotenv()

st.set_page_config(
    page_title="ResumeMatch Pro",
    page_icon="📄",
    layout="wide"
)

APP_TITLE = "ResumeMatch Pro"
APP_SUBTITLE = "AI-Powered ATS Resume Analyzer"


def extract_pdf_text(uploaded_file):
    """Extract text from an uploaded PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        return "\n\n".join(page.page_content for page in pages)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def extract_score(response_text):
    """Extract ATS score from LLM response."""
    match = re.search(r"ATS Score:\s*(\d{1,3})\s*/\s*100", response_text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(0, min(score, 100))
    return None


def get_score_label(score):
    if score >= 85:
        return "Excellent Match"
    elif score >= 70:
        return "Good Match"
    elif score >= 55:
        return "Average Match"
    return "Needs Improvement"


def main():
    st.title(f"📄 {APP_TITLE}")
    st.subheader(APP_SUBTITLE)

    st.markdown(
        """
        Upload your **resume PDF** and paste or upload a **job description** to get an AI-powered ATS match analysis,
        missing keywords, strengths, weaknesses, and personalized improvement suggestions.
        """
    )

    with st.sidebar:
        st.header("⚙️ Configuration")

        nvidia_api_key = st.text_input(
            "NVIDIA API Key",
            value=os.getenv("NVIDIA_API_KEY", ""),
            type="password",
            help="Enter your NVIDIA API key or store it in a .env file."
        )

        llm_model = st.text_input(
            "NVIDIA LLM Model",
            value=os.getenv("NVIDIA_LLM_MODEL", ""),
            help="Example: your preferred NVIDIA hosted LLM model name."
        )

        temperature = st.slider(
            "Creativity Level",
            min_value=0.0,
            max_value=1.0,
            value=0.33,
            step=0.01
        )

        st.divider()

        st.markdown("### Features")
        st.markdown(
            """
            ✅ ATS score  
            ✅ Skill matching  
            ✅ Missing keyword detection  
            ✅ Resume improvement suggestions  
            ✅ AI-generated final verdict  
            ✅ Downloadable report  
            """
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. Upload Resume")
        resume_file = st.file_uploader(
            "Upload your resume in PDF format",
            type=["pdf"]
        )

    with col2:
        st.markdown("### 2. Add Job Description")

        jd_input_method = st.radio(
            "Choose input method",
            ["Paste Job Description", "Upload TXT File"],
            horizontal=True
        )

        jd_text = ""

        if jd_input_method == "Paste Job Description":
            jd_text = st.text_area(
                "Paste the job description here",
                height=300,
                placeholder="Paste the complete job description..."
            )
        else:
            jd_file = st.file_uploader(
                "Upload job description as TXT file",
                type=["txt"]
            )
            if jd_file:
                jd_text = jd_file.read().decode("utf-8")

    analyze_button = st.button("🚀 Analyze Resume", use_container_width=True)

    if analyze_button:
        if not nvidia_api_key:
            st.error("Please provide your NVIDIA API key.")
            return

        if not llm_model:
            st.error("Please provide the NVIDIA LLM model name.")
            return

        if not resume_file:
            st.error("Please upload your resume PDF.")
            return

        if not jd_text.strip():
            st.error("Please provide the job description.")
            return

        with st.spinner("Extracting resume text..."):
            resume_text = extract_pdf_text(resume_file)

        if not resume_text.strip():
            st.error("Could not extract text from the resume PDF.")
            return

        with st.spinner("Analyzing resume against job description..."):
            full_response = analyze_resume(
                resume_text=resume_text,
                jd_text=jd_text,
                llm_model=llm_model,
                api_key=nvidia_api_key,
                temperature=temperature
            )

        st.divider()
        st.markdown("## 📊 ATS Analysis Report")
        st.markdown(full_response)

        score = extract_score(full_response)
        if score is not None:
            st.divider()
            st.markdown("## 🎯 Match Summary")

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("ATS Match Score", f"{score}/100")
            with metric_col2:
                st.metric("Result", get_score_label(score))

            st.progress(score / 100)

        st.divider()
        st.download_button(
            label="⬇️ Download ATS Report",
            data=full_response,
            file_name="ats_resume_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )


if __name__ == "__main__":
    main()