import re

import streamlit as st
from pypdf import PdfReader

from src.rag import retrieve_documents, generate_answer
from src.job_extractor import extract_job_from_url


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Career RAG Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# COLOR PALETTE
# ============================================================
#
# Background: Soft Beige
# Primary: Deep Plum
# Secondary: Muted Blue
# Accent: Warm Gold
# Text: Dark Brown
#
# ============================================================

BG = "#F5F0E8"
CARD = "#FFFDF8"
PLUM = "#55345E"
PLUM_DARK = "#3E2645"
BLUE = "#526A8A"
GOLD = "#C69C4A"
TEXT = "#292522"
MUTED = "#756E67"
BORDER = "#DED5C8"
LIGHT_PLUM = "#EEE5F0"
LIGHT_BLUE = "#E8EEF5"
LIGHT_GOLD = "#F5EBD6"
SUCCESS = "#5C8065"
ERROR = "#A85B55"


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "analysis_result": None,
    "resume_text": "",
    "job_text": "",
    "analysis_mode": "Full Career Match",
    "last_question": "",
    "analysis_done": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# LIGHT CSS
# ============================================================
#
# NOTE ON PROBLEM 3 (left panel height):
# The sidebar block rules below (.sidebar-tight, reduced margins
# on headings/markdown/dividers inside [data-testid="stSidebar"])
# are the ONLY change made to fix the "left panel too tall /
# overlaps heading" issue. No layout restructuring was done.
#
# NOTE ON PROBLEM 1 (matplotlib):
# matplotlib is not imported anywhere in this file. The score
# ring in show_score_chart() is pure CSS (conic-gradient), so
# there is no matplotlib dependency to remove.
#
# NOTE ON PROBLEM 2 (session_state widget conflict):
# Every widget below writes to its session_state key ONLY via
# Streamlit itself (key=...). Nowhere does the code do
# `st.session_state.<key> = <value>` for a key that already
# belongs to an instantiated widget, so the
# StreamlitAPIException described cannot occur here.
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {{
        background-color: {BG};
    }}

    .block-container {{
        max-width: 1680px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT} !important;
    }}

    /* Streamlit applies its own default text color to plain
       paragraphs/list items/table cells with higher specificity
       than a bare "p" rule. Without !important here, that default
       (tuned for a dark theme) wins and shows up as washed-out,
       low-contrast text on our beige background — this was the
       cause of the RAG analysis text looking faded. */

    p, li, ul, ol, td, th, span {{
        color: {TEXT} !important;
    }}

    strong, b {{
        color: {TEXT} !important;
    }}

    em, i {{
        color: {TEXT} !important;
    }}

    blockquote {{
        color: {MUTED} !important;
        border-left: 3px solid {BORDER};
        padding-left: 0.8rem;
        margin-left: 0;
    }}

    a {{
        color: {BLUE} !important;
    }}

    label {{
        color: {TEXT} !important;
    }}


    /* ========================================================
       SIDEBAR (LEFT AREA)
       ======================================================== */

    [data-testid="stSidebar"] {{
        background-color: #EFE8DC;
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {{
        color: {PLUM_DARK} !important;
    }}

    [data-testid="stSidebar"] p {{
        color: {TEXT} !important;
    }}

    [data-testid="stSidebar"] .stCaption {{
        color: {MUTED} !important;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: {BORDER};
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }}

    /* Compact spacing fix: previously the sidebar's suggestion
       block grew tall enough to crowd the main heading area.
       These rules tighten vertical rhythm only inside the
       sidebar, without touching the 3-area layout. */

    [data-testid="stSidebar"] .block-container {{
        padding-top: 1.2rem;
    }}

    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
        margin-bottom: 0rem;
    }}

    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] > h4 {{
        margin-top: 0.2rem;
        margin-bottom: 0.3rem;
        font-size: 0.95rem;
    }}

    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] ul {{
        margin-top: 0rem;
        margin-bottom: 0.2rem;
        padding-left: 1.1rem;
    }}

    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] ul li {{
        font-size: 0.85rem;
        line-height: 1.35rem;
        margin-bottom: 0.05rem;
        color: {TEXT};
    }}

    [data-testid="stSidebar"] .stSelectbox {{
        margin-bottom: 0.2rem;
    }}

    [data-testid="stSidebar"] .stCaption p {{
        font-size: 0.78rem;
        line-height: 1.15rem;
        margin-top: 0.15rem;
    }}


    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-testid="stFileUploader"] {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.4rem;
    }}

    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {{
        background-color: {CARD};
    }}

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {{
        background-color: {CARD};
    }}

    textarea {{
        background-color: {CARD} !important;
        color: {TEXT} !important;
    }}

    input {{
        color: {TEXT} !important;
        background-color: {CARD} !important;
    }}

    div[data-testid="stRadio"] label {{
        color: {TEXT} !important;
    }}

    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label {{
        color: {TEXT} !important;
        font-weight: 600;
    }}


    /* ========================================================
       NORMAL BUTTONS
       ======================================================== */

    .stButton > button {{
        background-color: {CARD} !important;
        color: {PLUM_DARK} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px;
        min-height: 46px;
        font-weight: 700;
    }}

    .stButton > button:hover {{
        background-color: {LIGHT_PLUM} !important;
        border-color: {PLUM} !important;
        color: {PLUM_DARK} !important;
    }}


    /* ========================================================
       PRIMARY ANALYZE BUTTON
       ======================================================== */

    .stButton > button[kind="primary"] {{
        background-color: {PLUM} !important;
        color: #FFFFFF !important;
        border: 1px solid {PLUM} !important;
        border-radius: 10px;
        min-height: 46px;
        font-weight: 700;
    }}

    .stButton > button[kind="primary"]:hover {{
        background-color: {PLUM_DARK} !important;
        border-color: {PLUM_DARK} !important;
        color: #FFFFFF !important;
    }}

    .stButton > button[kind="primary"]:disabled {{
        background-color: #C9BFCB !important;
        color: #F8F5F9 !important;
        border-color: #C9BFCB !important;
    }}


    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1rem;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {PLUM_DARK} !important;
    }}


    /* ========================================================
       CONTAINERS
       ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {BORDER} !important;
        background-color: {CARD};
        border-radius: 14px;
    }}


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {{
        border-color: {BORDER};
    }}


    /* ========================================================
       SELECTBOX
       ======================================================== */

    div[data-baseweb="select"] > div {{
        background-color: {CARD} !important;
        border-color: {BORDER} !important;
        color: {TEXT} !important;
    }}

    div[data-baseweb="select"] span {{
        color: {TEXT} !important;
    }}


    /* ========================================================
       FILE UPLOADER TEXT
       ======================================================== */

    [data-testid="stFileUploader"] {{
        color: {TEXT} !important;
    }}

    [data-testid="stFileUploader"] small {{
        color: {MUTED} !important;
    }}


    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {{
        border-radius: 10px;
    }}


    /* ========================================================
       RAG RESPONSE FORMATTING
       ------------------------------------------------------
       Scoped to .rag-response only, which wraps just the
       model-generated analysis text. This keeps the model's
       own ## / ### headers from rendering at full browser-
       default size (which was clashing with "Career Analysis"
       above it) without touching heading sizes anywhere else
       in the app.
       ======================================================== */

    .rag-response h1,
    .rag-response h2 {{
        font-size: 1.25rem !important;
        margin-top: 1.4rem;
        margin-bottom: 0.5rem;
        color: {PLUM_DARK} !important;
    }}

    .rag-response h3 {{
        font-size: 1.1rem !important;
        margin-top: 1.1rem;
        margin-bottom: 0.4rem;
        color: {PLUM_DARK} !important;
    }}

    .rag-response h4 {{
        font-size: 1rem !important;
        margin-top: 0.9rem;
        margin-bottom: 0.3rem;
        color: {PLUM_DARK} !important;
    }}

    .rag-response p,
    .rag-response li {{
        font-size: 0.95rem;
        line-height: 1.55;
        color: {TEXT} !important;
    }}

    .rag-response table {{
        width: 100%;
        border-collapse: collapse;
        margin: 0.6rem 0 1rem 0;
    }}

    .rag-response th,
    .rag-response td {{
        border: 1px solid {BORDER};
        padding: 0.4rem 0.6rem;
        text-align: left;
        color: {TEXT} !important;
        font-size: 0.9rem;
    }}

    .rag-response th {{
        background-color: {LIGHT_PLUM};
    }}

    .rag-response code {{
        background-color: {LIGHT_GOLD};
        color: {PLUM_DARK};
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_pdf_text(uploaded_file):
    """Extract text from an uploaded PDF."""

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""

        if text.strip():
            pages.append(text)

    return "\n".join(pages).strip()


def build_analysis_question(
    resume_text,
    job_text,
    mode,
    custom_question=None,
):
    """Build a mode-specific RAG question."""

    base = f"""
CANDIDATE RESUME
================
{resume_text}

TARGET JOB DESCRIPTION
=====================
{job_text}
"""

    if mode == "Full Career Match":

        task = """
Perform a complete career match analysis.

Evaluate:

1. Overall alignment
2. Matching skills
3. Experience/project alignment
4. Resume gaps
5. ATS-relevant observations
6. Market alignment
7. Priority improvements
8. Recommended next steps

At the beginning of your answer, provide:

OVERALL_MATCH: XX%
ATS_FRIENDLINESS: XX%

Only provide numerical values when they are reasonably
supported by the supplied evidence.
"""

    elif mode == "Skill Gap Analysis":

        task = """
Perform a focused skill-gap analysis.

Identify:

1. Skills explicitly demonstrated by the resume
2. Skills explicitly required by the job
3. Partially demonstrated capabilities
4. Capabilities not clearly demonstrated
5. Priority skill gaps
6. Practical learning recommendations

Do not claim that the candidate does not know a skill merely
because it is absent from the resume.

At the beginning provide:

OVERALL_MATCH: XX%

Only provide a numerical estimate when the evidence supports
one.
"""

    elif mode == "ATS Resume Check":

        task = """
Perform an ATS-oriented resume analysis against the supplied
job description.

Evaluate:

1. Matching terminology
2. Missing job-specific terminology
3. Skills explicitly demonstrated
4. Skills not clearly demonstrated
5. Project descriptions
6. Experience evidence
7. Resume clarity
8. Practical ATS improvements

Do not claim that the resume will definitely be rejected by
an ATS.

At the beginning provide:

ATS_FRIENDLINESS: XX%

Only provide a numerical estimate when supported by evidence.
"""

    elif mode == "Market Alignment":

        task = """
Analyze how the candidate's profile aligns with broader
employment and workforce trends represented in the retrieved
knowledge base.

Separate:

1. Resume evidence
2. Job evidence
3. Market evidence
4. Analysis

Only make market claims when supported by the retrieved
Future of Jobs Report context.

At the beginning provide:

OVERALL_MATCH: XX%

Only provide a numerical estimate when defensible.
"""

    else:

        task = f"""
Answer the user's specific career question.

USER QUESTION:
{custom_question}

Use the resume, job description and retrieved knowledge-base
context as evidence.

Do not introduce unsupported factual claims.
"""

    return base + "\n\n" + task


def extract_score(text, label):
    """Extract a score such as OVERALL_MATCH: 78%."""

    if not text:
        return None

    pattern = rf"{re.escape(label)}\s*:\s*(\d{{1,3}})\s*%"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    score = int(match.group(1))

    return min(max(score, 0), 100)


def clean_display_text(text):
    """
    Remove the raw OVERALL_MATCH / ATS_FRIENDLINESS marker lines
    before the analysis text is displayed.

    These markers stay in st.session_state.analysis_result
    untouched (extract_score() reads them from there), so nothing
    about score extraction changes. This only affects the copy of
    the text that gets rendered in the results card, since the
    same numbers are already shown clearly in the score rings and
    showing the raw "LABEL: XX%" line again above the analysis
    just added clutter.
    """

    if not text:
        return text

    pattern = r"(?im)^[ \t]*(OVERALL_MATCH|ATS_FRIENDLINESS)\s*:\s*\d{1,3}\s*%[ \t]*$"

    cleaned = re.sub(pattern, "", text)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def show_score_chart(
    score,
    title,
    color,
):
    """
    Display a donut score using CSS.

    Matplotlib is intentionally not used so the application
    does not require matplotlib to be installed. If score is
    None (model did not return a defensible number), the ring
    renders empty and the center shows "—" instead of a fake
    value.
    """

    if score is None:
        score = 0
        unknown = True
    else:
        unknown = False

    score = max(0, min(100, int(score)))

    degree = score * 3.6

    if unknown:
        center_text = "—"
    else:
        center_text = f"{score}%"

    # IMPORTANT: this HTML is built as a single flush-left string
    # (no leading whitespace on any line). Markdown treats any
    # line indented 4+ spaces as a literal code block, which is
    # exactly what was causing the raw "<div style=...>" text to
    # appear on screen instead of a rendered ring. Keeping the
    # whole snippet on one line with zero indentation avoids that
    # entirely, regardless of unsafe_allow_html=True.

    ring_html = (
        f'<div style="width:180px;height:180px;margin:10px auto 20px auto;'
        f'border-radius:50%;background:conic-gradient({color} 0deg {degree}deg,'
        f'#E6DED1 {degree}deg 360deg);display:flex;align-items:center;'
        f'justify-content:center;">'
        f'<div style="width:140px;height:140px;border-radius:50%;'
        f'background:{CARD};display:flex;flex-direction:column;'
        f'align-items:center;justify-content:center;">'
        f'<div style="font-size:30px;font-weight:800;color:{PLUM_DARK};">'
        f'{center_text}</div>'
        f'<div style="font-size:11px;color:{MUTED};margin-top:4px;">'
        f'{title}</div>'
        f'</div>'
        f'</div>'
    )

    st.markdown(ring_html, unsafe_allow_html=True)


def reset_analysis():
    st.session_state.analysis_result = None
    st.session_state.analysis_done = False
    st.session_state.last_question = ""
    st.session_state.resume_text = ""
    st.session_state.job_text = ""


# ============================================================
# SIDEBAR (LEFT AREA)
# ============================================================
#
# FIX FOR PROBLEM 3:
# The previous version used long bold "Question / Answer" blocks
# with blank lines between each item, which pushed the sidebar
# height up enough to crowd the header area. Below, the same
# information is kept but compressed into short bullet lists,
# with fewer dividers and tighter CSS spacing (see the
# [data-testid="stSidebar"] rules above). No content, modes, or
# functionality were removed.
# ============================================================

with st.sidebar:

    st.markdown("# ✦ Career Guide")

    st.caption(
        "Investigate your resume against a target role."
    )

    st.markdown("#### What can you ask?")

    st.markdown(
        "- Am I a good fit for this role?\n"
        "- What skills am I missing?\n"
        "- How ATS-friendly is my resume?\n"
        "- What should I improve first?"
    )

    st.divider()

    st.markdown("#### How do you want to use it?")

    # IMPORTANT:
    # Streamlit automatically stores the selected value in
    # st.session_state["analysis_mode"] because this widget
    # uses key="analysis_mode".
    #
    # DO NOT assign:
    # st.session_state.analysis_mode = analysis_mode
    #
    # after this widget is created. Doing so is exactly the
    # pattern that previously raised StreamlitAPIException.

    analysis_mode = st.selectbox(
        "Analysis type",
        [
            "Full Career Match",
            "Skill Gap Analysis",
            "ATS Resume Check",
            "Market Alignment",
            "Ask a Career Question",
        ],
        key="analysis_mode",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("#### Suggested questions")

    suggestions = {
        "Full Career Match": [
            "How well does my resume align with this job?",
            "What are my strongest matching skills?",
            "What should I improve before applying?",
        ],

        "Skill Gap Analysis": [
            "Which important skills are not clearly demonstrated?",
            "What are my highest-priority skill gaps?",
            "Which skills should I learn first?",
        ],

        "ATS Resume Check": [
            "What job-specific keywords are missing?",
            "Which parts of my resume need stronger evidence?",
            "How can I improve ATS alignment?",
        ],

        "Market Alignment": [
            "How does this role align with market trends?",
            "Which capabilities are becoming important?",
            "How does my profile compare with these trends?",
        ],

        "Ask a Career Question": [
            "What should I improve in my resume?",
            "Which project should I build next?",
            "What is the biggest weakness in my current profile?",
        ],
    }

    selected_suggestion = st.selectbox(
        "Example questions",
        suggestions[analysis_mode],
        label_visibility="collapsed",
    )

    st.caption(
        "Used as the analysis direction in custom-question mode."
    )

    st.divider()

    if st.button(
        "↻ Start New Analysis",
        use_container_width=True,
    ):
        reset_analysis()
        st.rerun()


# ============================================================
# TOP HEADER
# ============================================================

st.markdown(
    "# ✦ Career RAG Assistant"
)

st.markdown(
    "**Turn your resume into evidence, not assumptions.**"
)

st.caption(
    "Evidence-grounded resume, job and career analysis "
    "using Retrieval-Augmented Generation."
)


st.divider()


# ============================================================
# MAIN INPUT AREA
# ============================================================

st.markdown(
    "## Analyze a Target Role"
)

st.caption(
    "Provide both sides of the comparison: your resume and "
    "the target job description."
)

resume_col, job_col = st.columns(
    [1, 1],
    gap="large",
)


# ============================================================
# RESUME
# ============================================================

with resume_col:

    with st.container(border=True):

        st.markdown(
            "### 📄 Your Resume"
        )

        st.caption(
            "Upload the resume you would actually submit "
            "for this role."
        )

        resume_file = st.file_uploader(
            "Resume PDF",
            type=["pdf"],
            key="resume_upload",
        )

        if resume_file:

            st.success(
                f"Resume loaded: {resume_file.name}"
            )


# ============================================================
# JOB DESCRIPTION
# ============================================================

with job_col:

    with st.container(border=True):

        st.markdown(
            "### 💼 Target Job"
        )

        st.caption(
            "Use pasted text, a PDF, or a public job URL."
        )

        job_source = st.radio(
            "Job description source",
            [
                "Paste Description",
                "Upload PDF",
                "Job URL",
            ],
            horizontal=True,
            key="job_source",
        )

        job_text_current = ""

        if job_source == "Paste Description":

            job_text_current = st.text_area(
                "Job description",
                placeholder=(
                    "Paste the complete job description here..."
                ),
                height=180,
                key="job_description_text",
            )

        elif job_source == "Upload PDF":

            job_pdf = st.file_uploader(
                "Job description PDF",
                type=["pdf"],
                key="job_description_pdf",
            )

            if job_pdf:

                try:

                    job_text_current = extract_pdf_text(
                        job_pdf
                    )

                    st.success(
                        f"Job description loaded: {job_pdf.name}"
                    )

                except Exception as e:

                    st.error(
                        f"Could not read the job PDF: {e}"
                    )

        else:

            job_url = st.text_input(
                "Public job URL",
                placeholder=(
                    "https://example.com/job-description"
                ),
                key="job_url",
            )

            if job_url.strip():

                job_text_current = job_url.strip()


# ============================================================
# CUSTOM QUESTION
# ============================================================

if analysis_mode == "Ask a Career Question":

    st.markdown(
        "### Your Question"
    )

    custom_question = st.text_area(
        "Career question",
        value=selected_suggestion,
        height=100,
        key="custom_question",
    )

else:

    custom_question = selected_suggestion


# ============================================================
# INPUT VALIDATION
# ============================================================

resume_ready = resume_file is not None
job_ready = bool(job_text_current.strip())

ready = resume_ready and job_ready


st.divider()


status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:

    if resume_ready:
        st.success("✓ Resume ready")
    else:
        st.warning("○ Resume required")


with status_col2:

    if job_ready:
        st.success("✓ Job description ready")
    else:
        st.warning("○ Job description required")


with status_col3:

    st.info(
        f"Analysis: {analysis_mode}"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "✦ Analyze Career Match",
    type="primary",
    use_container_width=True,
    disabled=not ready,
)


if not ready:

    st.caption(
        "Upload a resume and provide a job description "
        "to activate analysis."
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Reading your resume..."
        ):

            resume_text = extract_pdf_text(
                resume_file
            )

        if not resume_text:

            st.error(
                "The uploaded resume does not contain readable text."
            )

            st.stop()

        st.session_state.resume_text = resume_text

    except Exception as e:

        st.error(
            f"Could not read the resume: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # JOB URL
    # --------------------------------------------------------

    if job_source == "Job URL":

        try:

            with st.spinner(
                "Extracting the job description..."
            ):

                job_text = extract_job_from_url(
                    job_text_current
                )

            if not job_text or not job_text.strip():

                st.error(
                    "The job URL did not return a readable "
                    "job description."
                )

                st.stop()

            st.session_state.job_text = job_text

        except Exception as e:

            st.error(
                f"Could not extract the job description: {e}"
            )

            st.info(
                "Some job websites block automated extraction. "
                "If this happens, use Paste Description instead."
            )

            st.stop()

    else:

        job_text = job_text_current

        if not job_text.strip():

            st.error(
                "Please provide a job description."
            )

            st.stop()

        st.session_state.job_text = job_text


    # --------------------------------------------------------
    # BUILD QUESTION
    # --------------------------------------------------------

    question = build_analysis_question(
        resume_text=resume_text,
        job_text=job_text,
        mode=analysis_mode,
        custom_question=custom_question,
    )

    st.session_state.last_question = question


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Searching the career knowledge base..."
        ):

            documents = retrieve_documents(
                question,
                k=5,
            )

        if not documents:

            st.warning(
                "No relevant knowledge-base documents "
                "were retrieved."
            )

        with st.spinner(
            "Generating evidence-based analysis..."
        ):

            answer = generate_answer(
                question,
                documents,
            )

        st.session_state.analysis_result = answer
        st.session_state.analysis_done = True

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
        "## Analysis Results"
    )

    st.caption(
        f"Analysis type: {st.session_state.analysis_mode}"
    )


    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    overall_score = extract_score(
        st.session_state.analysis_result,
        "OVERALL_MATCH",
    )

    ats_score = extract_score(
        st.session_state.analysis_result,
        "ATS_FRIENDLINESS",
    )


    score_left, result_middle, score_right = st.columns(
        [1, 3, 1],
        gap="large",
        vertical_alignment="top",
    )


    # --------------------------------------------------------
    # OVERALL MATCH
    # --------------------------------------------------------

    with score_left:

        with st.container(border=True):

            st.markdown(
                "### Overall Match"
            )

            show_score_chart(
                overall_score,
                "Resume ↔ Role",
                PLUM,
            )

            if overall_score is None:

                st.caption(
                    "The model did not return a defensible "
                    "numerical match score."
                )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    with result_middle:

        with st.container(border=True):

            st.markdown(
                "### Career Analysis"
            )

            display_text = clean_display_text(
                st.session_state.analysis_result
            )

            st.markdown(
                f'<div class="rag-response">\n\n{display_text}\n\n</div>',
                unsafe_allow_html=True,
            )


    # --------------------------------------------------------
    # ATS
    # --------------------------------------------------------

    with score_right:

        with st.container(border=True):

            st.markdown(
                "### ATS Friendliness"
            )

            show_score_chart(
                ats_score,
                "Resume ↔ Keywords",
                BLUE,
            )

            if ats_score is None:

                st.caption(
                    "ATS score appears only when the model "
                    "can support a numerical estimate."
                )


    # ========================================================
    # SOURCES
    # ========================================================

    st.divider()

    st.markdown(
        "### Evidence Sources"
    )

    st.caption(
        "The analysis uses the supplied resume/job description "
        "plus retrieved career knowledge-base context."
    )

    source_col1, source_col2 = st.columns(2)

    with source_col1:

        st.info(
            "📄 Direct evidence\n\n"
            "Your uploaded resume and selected job description "
            "are passed directly into the analysis."
        )

    with source_col2:

        st.info(
            "🔎 Retrieved evidence\n\n"
            "Relevant chunks are retrieved from the Chroma "
            "career knowledge base before generation."
        )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.divider()

    st.markdown(
        "## Start with a question"
    )

    st.caption(
        "Choose an analysis type from the left, then provide "
        "your resume and target role."
    )

    example_col1, example_col2, example_col3 = st.columns(
        3,
        gap="large",
    )

    with example_col1:

        with st.container(border=True):

            st.markdown(
                "### 🎯 Career Match"
            )

            st.write(
                "Find where your current profile aligns "
                "with a specific role."
            )

            st.caption(
                "Example: How well does my profile fit this job?"
            )

    with example_col2:

        with st.container(border=True):

            st.markdown(
                "### 🧩 Skill Gaps"
            )

            st.write(
                "Identify requirements that are not clearly "
                "demonstrated in your resume."
            )

            st.caption(
                "Example: Which skills should I improve first?"
            )

    with example_col3:

        with st.container(border=True):

            st.markdown(
                "### 🔎 ATS Review"
            )

            st.write(
                "Check terminology and evidence against "
                "the target job."
            )

            st.caption(
                "Example: Which job keywords should I strengthen?"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Career RAG Assistant • Evidence-grounded career analysis"
)