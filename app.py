import streamlit as st
from pypdf import PdfReader

from src.rag import retrieve_documents, generate_answer
from src.job_extract import extract_job_from_url


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Career RAG Assistant",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #F8F9FA;
    }

    .block-container {
        max-width: 1420px;
        padding: 28px 42px 50px;
    }

    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }

    h1, h2, h3 {
        color: #111827 !important;
    }

    p, label {
        color: #6B7280 !important;
    }

    /* TOP BAR */

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 22px;
        margin-bottom: 30px;
        border-bottom: 1px solid #E5E7EB;
    }

    .brand-area {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563EB, #6D28D9);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 12px;
    }

    .brand-name {
        color: #111827;
        font-size: 18px;
        font-weight: 800;
        line-height: 1.1;
    }

    .brand-desc {
        color: #6B7280;
        font-size: 11px;
        margin-top: 4px;
    }

    /* CARDS */

    .card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 22px;
    }

    /* STATUS */

    .status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 10px;
        border-radius: 999px;
        background: #F3F4F6;
        color: #6B7280;
        font-size: 11px;
        font-weight: 600;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #9CA3AF;
    }

    .status-ready .status-dot {
        background: #22C55E;
    }

    /* INFO PANEL */

    .info-title {
        font-size: 13px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 6px;
    }

    .info-text {
        font-size: 12px;
        line-height: 1.6;
        color: #6B7280;
    }

    /* BUTTON */

    div.stButton > button {
        width: 100%;
        height: 42px;
        border-radius: 9px;
        border: none;
        background: #2563EB;
        color: white;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background: #1D4ED8;
        color: white;
    }

    /* INPUTS */

    .stTextArea textarea,
    .stTextInput input {
        background: white !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 9px !important;
        color: #111827 !important;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #CBD5E1;
        border-radius: 10px;
    }

    /* RESULT CARDS */

    .result-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 18px;
        min-height: 110px;
    }

    .result-label {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .08em;
        color: #6D28D9;
        margin-bottom: 10px;
    }

    .result-text {
        font-size: 13px;
        color: #374151;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "job_text" not in st.session_state:
    st.session_state.job_text = ""


# ============================================================
# TOP BAR
# ============================================================

top_left, top_center, top_right = st.columns(
    [1.8, 1.4, 0.7],
    vertical_alignment="center"
)

with top_left:

    st.markdown(
        """
        <div class="brand-area">
            <div class="brand-icon">AI</div>

            <div>
                <div class="brand-name">
                    Career RAG Assistant
                </div>

                <div class="brand-desc">
                    AI-powered Resume & Career Analysis
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with top_center:

    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        st.button("Dashboard", disabled=True)

    with nav2:
        st.button("Analysis", disabled=True)

    with nav3:
        st.button("AI Chat", disabled=True)


with top_right:

    st.markdown(
        """
        <div class="status">
            <span class="status-dot"></span>
            RAG Ready
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# INTRODUCTION
# ============================================================

left, right = st.columns(
    [1, 2.3],
    gap="large"
)


with left:

    st.markdown(
        """
        <div class="card">

            <h3>Career RAG Assistant</h3>

            <div class="info-text">
                Analyze a resume against a target job description
                using Retrieval-Augmented Generation.
            </div>

            <br>

            <div class="info-title">
                Knowledge Base
            </div>

            <div class="info-text">
                Resume profile + World Economic Forum
                Future of Jobs Report 2025.
            </div>

            <br>

            <div class="info-title">
                Analysis
            </div>

            <div class="info-text">
                The system retrieves relevant career evidence
                before generating the final analysis.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with right:

    st.subheader("Career Match Analysis")

    st.caption(
        "Compare your resume against a specific job description."
    )

    resume_col, job_col = st.columns(
        2,
        gap="large"
    )


    # ========================================================
    # RESUME
    # ========================================================

    with resume_col:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("### Resume")

        st.caption(
            "Upload your current resume as a PDF."
        )

        resume_file = st.file_uploader(
            "Resume PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )

        st.caption("PDF format")

        st.markdown("</div>", unsafe_allow_html=True)


    # ========================================================
    # JOB DESCRIPTION
    # ========================================================

    with job_col:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("### Job Description")

        st.caption(
            "Paste text, upload a PDF, or provide a public URL."
        )

        job_source = st.radio(
            "Job description source",
            ["Text", "PDF", "URL"],
            horizontal=True,
            label_visibility="collapsed",
        )

        job_text = ""
        job_pdf = None
        job_url = ""


        if job_source == "Text":

            job_text = st.text_area(
                "Job description",
                placeholder=(
                    "Paste the job description here..."
                ),
                height=145,
                label_visibility="collapsed",
            )


        elif job_source == "PDF":

            job_pdf = st.file_uploader(
                "Job description PDF",
                type=["pdf"],
                key="job_description_pdf",
                label_visibility="collapsed",
            )


        else:

            job_url = st.text_input(
                "Job URL",
                placeholder="https://...",
                label_visibility="collapsed",
            )

        st.markdown("</div>", unsafe_allow_html=True)


    st.write("")

    analyze_clicked = st.button(
        "Analyze Career Match",
        type="primary",
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze_clicked:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if resume_file is None:

        st.error("Please upload your resume.")

        st.stop()


    if job_source == "Text" and not job_text.strip():

        st.error("Please enter the job description.")

        st.stop()


    if job_source == "PDF" and job_pdf is None:

        st.error(
            "Please upload the job description PDF."
        )

        st.stop()


    if job_source == "URL" and not job_url.strip():

        st.error("Please enter the job URL.")

        st.stop()


    # --------------------------------------------------------
    # PROCESSING
    # --------------------------------------------------------

    with st.spinner(
        "Extracting documents and running RAG analysis..."
    ):

        try:

            # ------------------------------------------------
            # Resume extraction
            # ------------------------------------------------

            resume_reader = PdfReader(resume_file)

            resume_text = "\n".join(
                page.extract_text() or ""
                for page in resume_reader.pages
            ).strip()


            if not resume_text:

                raise ValueError(
                    "Could not extract readable text "
                    "from the uploaded resume."
                )


            # ------------------------------------------------
            # Job PDF
            # ------------------------------------------------

            if job_source == "PDF":

                job_reader = PdfReader(job_pdf)

                job_text = "\n".join(
                    page.extract_text() or ""
                    for page in job_reader.pages
                ).strip()


                if not job_text:

                    raise ValueError(
                        "Could not extract readable text "
                        "from the job description PDF."
                    )


            # ------------------------------------------------
            # Job URL
            # ------------------------------------------------

            elif job_source == "URL":

                job_text = extract_job_from_url(
                    job_url.strip()
                )


            # ------------------------------------------------
            # Store input
            # ------------------------------------------------

            st.session_state.resume_text = resume_text
            st.session_state.job_text = job_text


            # ------------------------------------------------
            # Analysis question
            # ------------------------------------------------

            question = f"""
Analyze the candidate resume against the target job description.

CANDIDATE RESUME
================

{resume_text}


TARGET JOB DESCRIPTION
======================

{job_text}


TASK
====

Provide an evidence-based career match analysis.

Evaluate:

1. Matching skills
2. Skills or capabilities not clearly demonstrated
3. Experience alignment
4. ATS-relevant observations
5. Market alignment
6. Overall match assessment
7. Recommended improvements

Important:

The uploaded resume and job description are direct user-provided
evidence and may be used for the comparison.

Use the retrieved knowledge-base documents to support
career and market-related claims.

Do not assume that absence from the resume proves that
the candidate does not know a skill.
"""


            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            documents = retrieve_documents(
                question,
                k=5
            )


            # ------------------------------------------------
            # Generation
            # ------------------------------------------------

            answer = generate_answer(
                question,
                documents
            )


            st.session_state.answer = answer
            st.session_state.analysis_done = True


            st.rerun()


        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_done:

    st.divider()

    st.subheader("Analysis Results")

    st.caption(
        "Generated using the uploaded resume, target job "
        "description, and retrieved career knowledge."
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-label">
                    RESUME
                </div>

                <div class="result-text">
                    Resume successfully extracted and analyzed.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with summary_col2:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-label">
                    JOB DESCRIPTION
                </div>

                <div class="result-text">
                    Target job description successfully processed.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    st.write("")


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    st.subheader("AI Career Analysis")

    with st.container(border=True):

        st.markdown(
            st.session_state.answer
        )