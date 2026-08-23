import streamlit as st
from pathlib import Path
import tempfile

from pypdf import PdfReader

from src.rag import retrieve_documents, generate_answer
from src.job_extractor import extract_job_from_url


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Career RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #F8F9FA;
        color: #111827;
    }

    .block-container {
        max-width: 1400px;
        padding: 28px 42px 50px 42px;
    }

    header[data-testid="stHeader"] {
        background: #F8F9FA;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- HEADER ---------- */

    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 4px 0 22px 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 28px;
    }

    .brand-area {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(
            135deg,
            #6D28D9,
            #2563EB
        );
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 800;
    }

    .brand-title {
        font-size: 20px;
        font-weight: 800;
        line-height: 1.1;
        color: #111827;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #6B7280;
        margin-top: 4px;
    }

    .new-analysis {
        background: #2563EB;
        color: white;
        border-radius: 9px;
        padding: 10px 17px;
        font-size: 12px;
        font-weight: 700;
        text-align: center;
    }

    /* ---------- TITLES ---------- */

    .page-title {
        font-size: 25px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .page-subtitle {
        font-size: 13px;
        color: #6B7280;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .section-subtitle {
        font-size: 12px;
        color: #6B7280;
        margin-bottom: 18px;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 2px 8px rgba(17, 24, 39, 0.03);
    }

    /* ---------- SCORE ---------- */

    .score-card {
        min-height: 540px;
    }

    .score-heading {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .score-subtitle {
        font-size: 12px;
        color: #6B7280;
        line-height: 1.5;
        margin-bottom: 18px;
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #F3F4F6;
        border: 1px solid #E5E7EB;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 11px;
        color: #6B7280;
        margin-bottom: 30px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #9CA3AF;
        border-radius: 50%;
    }

    .score-heading-small {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .08em;
        color: #6B7280;
        margin-bottom: 14px;
    }

    .score-circle {
        width: 190px;
        height: 190px;
        border-radius: 50%;
        background:
            conic-gradient(
                #6D28D9 0deg,
                #6D28D9 0deg,
                #EDE9FE 0deg,
                #EDE9FE 360deg
            );
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 12px auto 25px auto;
    }

    .score-circle-inner {
        width: 154px;
        height: 154px;
        border-radius: 50%;
        background: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .score-number {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        line-height: 1;
    }

    .score-label {
        font-size: 11px;
        color: #6B7280;
        margin-top: 7px;
    }

    .score-description {
        text-align: center;
        color: #6B7280;
        font-size: 12px;
        line-height: 1.6;
        max-width: 280px;
        margin: auto;
    }

    /* ---------- INPUT CARDS ---------- */

    .input-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 15px;
        padding: 20px;
        min-height: 170px;
    }

    .input-title {
        font-size: 16px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .input-description {
        font-size: 12px;
        color: #6B7280;
        line-height: 1.5;
        margin-bottom: 15px;
    }

    .upload-note {
        font-size: 11px;
        color: #9CA3AF;
        margin-top: 7px;
    }

    /* ---------- RADIO ---------- */

    div[data-testid="stRadio"] {
        margin-top: -4px;
        margin-bottom: 8px;
    }

    div[data-testid="stRadio"] label {
        font-size: 12px !important;
    }

    /* ---------- TEXTAREA ---------- */

    textarea {
        border-radius: 10px !important;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        height: 46px;
        border-radius: 10px;
        border: none;
        background: #2563EB;
        color: white;
        font-size: 13px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #1D4ED8;
        color: white;
    }

    /* ---------- METRICS ---------- */

    .metric-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 18px;
        min-height: 115px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 800;
        color: #111827;
    }

    .metric-label {
        font-size: 11px;
        color: #6B7280;
        margin-top: 5px;
        margin-bottom: 10px;
    }

    .metric-bar {
        width: 100%;
        height: 6px;
        background: #EDE9FE;
        border-radius: 99px;
        overflow: hidden;
    }

    .metric-fill {
        height: 100%;
        background: #6D28D9;
        border-radius: 99px;
    }

    /* ---------- SKILLS ---------- */

    .skill-box {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 20px;
        min-height: 150px;
    }

    .skill-heading {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .07em;
        color: #6B7280;
        margin-bottom: 13px;
    }

    .skill-pill {
        display: inline-block;
        padding: 7px 10px;
        margin: 0 5px 7px 0;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
    }

    .skill-match {
        background: #EEF2FF;
        color: #4338CA;
    }

    .skill-gap {
        background: #F3F4F6;
        color: #6B7280;
    }

    /* ---------- INSIGHTS ---------- */

    .insight-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 18px;
        min-height: 105px;
    }

    .insight-label {
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .08em;
        color: #6D28D9;
        margin-bottom: 10px;
    }

    .insight-text {
        font-size: 12px;
        color: #374151;
        line-height: 1.5;
    }

    /* ---------- STEPS ---------- */

    .step-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 9px;
        font-size: 12px;
        color: #374151;
    }

    .step-number {
        color: #6D28D9;
        font-weight: 800;
        margin-right: 10px;
    }

    /* ---------- CHAT ---------- */

    .chat-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 20px;
    }

    .chat-label {
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .08em;
        color: #6D28D9;
    }

    .chat-title {
        font-size: 18px;
        font-weight: 800;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    .user-message {
        background: #2563EB;
        color: white;
        padding: 10px 13px;
        border-radius: 12px 12px 3px 12px;
        font-size: 12px;
        margin-left: 35px;
        margin-bottom: 12px;
    }

    .ai-message {
        background: #F3F4F6;
        color: #374151;
        padding: 11px 13px;
        border-radius: 12px 12px 12px 3px;
        font-size: 12px;
        line-height: 1.5;
        margin-right: 15px;
    }

    /* ---------- SOURCES ---------- */

    .source-card {
        background: #F3F4F6;
        border-radius: 12px;
        padding: 14px 16px;
    }

    .source-title {
        font-size: 12px;
        font-weight: 800;
        color: #374151;
        margin-bottom: 8px;
    }

    .source-item {
        font-size: 11px;
        color: #6B7280;
        margin-top: 6px;
    }

    /* ---------- ALERTS ---------- */

    .stAlert {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "job_text" not in st.session_state:
    st.session_state.job_text = ""

if "job_source" not in st.session_state:
    st.session_state.job_source = "Paste Text"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="brand-area">
            <div class="brand-icon">AI</div>
            <div>
                <div class="brand-title">Career RAG</div>
                <div class="brand-subtitle">
                    AI-powered Resume & Job Match Analysis
                </div>
            </div>
        </div>

        <div class="new-analysis">
            + New Analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT / SCORE AREA
# ============================================================

left, right = st.columns(
    [0.85, 1.65],
    gap="large"
)


# ============================================================
# LEFT — SCORE
# ============================================================

with left:

    st.markdown(
        """
        <div class="card score-card">

            <div class="score-heading">
                Your Career Match
            </div>

            <div class="score-subtitle">
                AI-powered resume and job description analysis.
            </div>

            <div class="status">
                <span class="status-dot"></span>
                Waiting for analysis
            </div>

            <div class="score-heading-small">
                MAIN SCORE
            </div>

            <div class="score-circle">

                <div class="score-circle-inner">

                    <div class="score-number">
                        0%
                    </div>

                    <div class="score-label">
                        Overall Match
                    </div>

                </div>

            </div>

            <div class="score-description">
                Upload your resume and provide a job description
                to calculate your match.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RIGHT — INPUTS
# ============================================================

with right:

    st.markdown(
        '<div class="page-title">Career Match Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Compare your resume against a specific job description.</div>',
        unsafe_allow_html=True,
    )

    resume_col, job_col = st.columns(2, gap="large")


    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    with resume_col:

        st.markdown(
            """
            <div class="input-card">
                <div class="input-title">Resume</div>
                <div class="input-description">
                    Upload your current resume.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        resume_file = st.file_uploader(
            "Upload resume",
            type=["pdf"],
            label_visibility="collapsed",
            key="resume_upload",
        )

        st.markdown(
            '<div class="upload-note">PDF • Maximum 200MB</div>',
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # JOB DESCRIPTION
    # --------------------------------------------------------

    with job_col:

        st.markdown(
            """
            <div class="input-card">
                <div class="input-title">Job Description</div>
                <div class="input-description">
                    Paste text, upload a PDF, or provide a job URL.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        job_source = st.radio(
            "Job description source",
            [
                "Paste Text",
                "Upload PDF",
                "Job URL",
            ],
            horizontal=True,
            key="job_source_radio",
        )

        st.session_state.job_source = job_source

        if job_source == "Paste Text":

            job_text = st.text_area(
                "Job description",
                placeholder="Paste the complete job description here...",
                height=145,
                label_visibility="collapsed",
                key="job_text_input",
            )

            st.session_state.job_text = job_text


        elif job_source == "Upload PDF":

            job_pdf = st.file_uploader(
                "Upload job description PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="job_pdf_upload",
            )

            if job_pdf:

                reader = PdfReader(job_pdf)

                pages = []

                for page in reader.pages:
                    pages.append(
                        page.extract_text() or ""
                    )

                st.session_state.job_text = "\n".join(pages)


        else:

            job_url = st.text_input(
                "Job URL",
                placeholder="https://example.com/job-description",
                label_visibility="collapsed",
                key="job_url_input",
            )

            if job_url:

                st.session_state.job_text = job_url


    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    st.write("")

    analyze = st.button(
        "Analyze Career Match",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if resume_file is None:

        st.error("Please upload your resume.")

        st.stop()


    # --------------------------------------------------------
    # READ RESUME
    # --------------------------------------------------------

    try:

        resume_reader = PdfReader(resume_file)

        resume_pages = []

        for page in resume_reader.pages:

            resume_pages.append(
                page.extract_text() or ""
            )

        resume_text = "\n".join(resume_pages)

        st.session_state.resume_text = resume_text

    except Exception as e:

        st.error(
            f"Could not read the resume: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # JOB TEXT
    # --------------------------------------------------------

    if job_source == "Job URL":

        url = st.session_state.job_text.strip()

        if not url:

            st.error("Please provide a job URL.")

            st.stop()

        try:

            with st.spinner("Extracting job description..."):

                job_text = extract_job_from_url(url)

            st.session_state.job_text = job_text

        except Exception as e:

            st.error(
                f"Could not extract the job description: {e}"
            )

            st.stop()


    else:

        job_text = st.session_state.job_text.strip()

        if not job_text:

            st.error(
                "Please provide a job description."
            )

            st.stop()


    # --------------------------------------------------------
    # BUILD QUESTION
    # --------------------------------------------------------

    question = f"""
Analyze the candidate against the target job description.

CANDIDATE RESUME
================

{resume_text}


TARGET JOB DESCRIPTION
======================

{job_text}


Provide:

1. Overall assessment
2. Matching skills
3. Experience alignment
4. Resume gaps
5. ATS-relevant observations
6. Market alignment
7. Priority improvements
8. Recommended next steps
"""


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Analyzing resume and job description..."
        ):

            documents = retrieve_documents(
                question,
                k=5,
            )

            answer = generate_answer(
                question,
                documents,
            )

        st.session_state.analysis_result = answer

    except Exception as e:

        st.error(
            f"Analysis failed: {e}"
        )

        st.stop()


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_result:

    st.divider()

    st.markdown(
        '<div class="page-title">Analysis Results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Evidence-based comparison of your resume and target role.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        st.session_state.analysis_result
    )