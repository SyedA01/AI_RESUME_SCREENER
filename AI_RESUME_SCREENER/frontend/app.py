import re
import pickle
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pypdf import PdfReader
from docx import Document


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# ROLE REQUIREMENTS
# ============================================================

ROLE_REQUIREMENTS = {

    "Data Analyst": [
        "python",
        "sql",
        "excel",
        "power bi",
        "tableau",
        "statistics",
        "pandas",
        "numpy",
        "data visualization",
        "machine learning"
    ],

    "Machine Learning Engineer": [
        "python",
        "machine learning",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "numpy",
        "pandas",
        "deep learning",
        "sql",
        "docker"
    ],

    "Full Stack Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "node.js",
        "express",
        "mongodb",
        "rest api",
        "git",
        "docker"
    ],

    "Business Analyst": [
        "excel",
        "sql",
        "power bi",
        "business analysis",
        "statistics",
        "data visualization",
        "requirements gathering",
        "documentation",
        "agile",
        "jira"
    ]
}


# ============================================================
# SKILL ALIASES
# ============================================================

SKILL_ALIASES = {

    "python": [
        "python",
        "python programming"
    ],

    "sql": [
        "sql",
        "structured query language"
    ],

    "excel": [
        "excel",
        "microsoft excel",
        "ms excel"
    ],

    "power bi": [
        "power bi",
        "powerbi",
        "microsoft power bi"
    ],

    "tableau": [
        "tableau"
    ],

    "statistics": [
        "statistics",
        "statistical analysis"
    ],

    "pandas": [
        "pandas"
    ],

    "numpy": [
        "numpy"
    ],

    "data visualization": [
        "data visualization",
        "data visualisation"
    ],

    "machine learning": [
        "machine learning",
        "machine-learning"
    ],

    "tensorflow": [
        "tensorflow"
    ],

    "pytorch": [
        "pytorch"
    ],

    "scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn"
    ],

    "deep learning": [
        "deep learning",
        "deep-learning"
    ],

    "docker": [
        "docker"
    ],

    "html": [
        "html",
        "html5"
    ],

    "css": [
        "css",
        "css3"
    ],

    "javascript": [
        "javascript",
        "java script",
        "js"
    ],

    "react": [
        "react",
        "react.js",
        "reactjs"
    ],

    "node.js": [
        "node.js",
        "nodejs",
        "node js"
    ],

    "express": [
        "express",
        "express.js",
        "expressjs"
    ],

    "mongodb": [
        "mongodb",
        "mongo db"
    ],

    "rest api": [
        "rest api",
        "restful api",
        "restful apis"
    ],

    "git": [
        "git"
    ],

    "business analysis": [
        "business analysis",
        "business analyst"
    ],

    "requirements gathering": [
        "requirements gathering",
        "requirement gathering",
        "requirements analysis"
    ],

    "documentation": [
        "documentation",
        "technical documentation"
    ],

    "agile": [
        "agile",
        "agile methodology"
    ],

    "jira": [
        "jira",
        "atlassian jira"
    ]
}


# ============================================================
# LOAD MODEL + TOKENIZER
# ============================================================

@st.cache_resource
def load_artifacts():

    # app.py is inside:
    # AI_RESUME_SCREENER/frontend/app.py

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    MODEL_PATH = PROJECT_ROOT / "notebooks" / "resume_match_model.keras"
    TOKENIZER_PATH = PROJECT_ROOT / "notebooks" / "tokenizer.pkl"

    # Debug information
    print("PROJECT ROOT:", PROJECT_ROOT)
    print("MODEL PATH:", MODEL_PATH)
    print("MODEL EXISTS:", MODEL_PATH.exists())
    print("TOKENIZER PATH:", TOKENIZER_PATH)
    print("TOKENIZER EXISTS:", TOKENIZER_PATH.exists())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at:\n{TOKENIZER_PATH}"
        )

    # Load model
    model = load_model(
        str(MODEL_PATH)
    )

    # Load tokenizer
    with open(
        TOKENIZER_PATH,
        "rb"
    ) as f:
        tokenizer = pickle.load(f)

    return model, tokenizer
# ============================================================
# DOCUMENT PARSING
# ============================================================

def extract_pdf_text(uploaded_file):

    uploaded_file.seek(0)

    reader = PdfReader(uploaded_file)

    text = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


def extract_docx_text(uploaded_file):

    uploaded_file.seek(0)

    document = Document(uploaded_file)

    text = []

    # Normal paragraphs
    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            text.append(
                paragraph.text
            )

    # Tables
    for table in document.tables:

        for row in table.rows:

            for cell in row.cells:

                if cell.text.strip():

                    text.append(
                        cell.text
                    )

    return "\n".join(text)


def extract_txt_text(uploaded_file):

    uploaded_file.seek(0)

    data = uploaded_file.read()

    try:

        return data.decode("utf-8")

    except UnicodeDecodeError:

        return data.decode(
            "latin-1",
            errors="ignore"
        )


def extract_document_text(uploaded_file):

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf_text(
            uploaded_file
        )

    elif filename.endswith(".docx"):

        return extract_docx_text(
            uploaded_file
        )

    elif filename.endswith(".txt"):

        return extract_txt_text(
            uploaded_file
        )

    else:

        raise ValueError(
            "Unsupported file type."
        )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(
    resume_text,
    selected_role
):

    text = clean_text(
        resume_text
    )

    required_skills = ROLE_REQUIREMENTS[
        selected_role
    ]

    detected_skills = []

    for skill in required_skills:

        aliases = SKILL_ALIASES.get(
            skill,
            [skill]
        )

        for alias in aliases:

            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(alias.lower())
                + r"(?![a-z0-9])"
            )

            if re.search(
                pattern,
                text
            ):

                detected_skills.append(
                    skill
                )

                break

    return sorted(
        set(detected_skills)
    )


# ============================================================
# SKILL MATCHING
# ============================================================

def calculate_skill_match(
    required_skills,
    candidate_skills
):

    required = set(
        required_skills
    )

    candidate = set(
        candidate_skills
    )

    matched = sorted(
        required.intersection(
            candidate
        )
    )

    missing = sorted(
        required.difference(
            candidate
        )
    )

    percentage = (
        len(matched)
        /
        len(required)
        *
        100
    )

    return (
        matched,
        missing,
        percentage
    )


# ============================================================
# LSTM PREDICTION
# ============================================================

def predict_with_lstm(
    model,
    tokenizer,
    role,
    required_skills,
    candidate_skills
):

    # Keep preprocessing consistent with
    # the training dataset.

    required_text = (
        "technical skills: "
        +
        ", ".join(
            required_skills
        )
    )

    candidate_text = (
        "candidate skills: "
        +
        ", ".join(
            candidate_skills
        )
    )

    model_text = (
        "job role "
        +
        role
        +
        " required "
        +
        required_text
        +
        " candidate "
        +
        candidate_text
    )

    sequence = tokenizer.texts_to_sequences(
        [model_text]
    )

    # MUST match training.
    MAX_LEN = 50

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = float(
        model.predict(
            padded,
            verbose=0
        )[0][0]
    )

    if probability >= 0.50:

        prediction = "Suitable"

    else:

        prediction = "Not Suitable"

    return (
        probability,
        prediction,
        model_text
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(
    skill_match
):

    if skill_match >= 80:

        return "Highly Suitable"

    elif skill_match >= 50:

        return "Moderately Suitable"

    else:

        return "Not Suitable"


# ============================================================
# HEADER
# ============================================================

st.title(
    "📄 AI-Powered Resume Screening System"
)

st.write(
    "Upload a resume and evaluate its suitability "
    "for a selected job role using NLP, skill matching "
    "and an LSTM model."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Screening Settings"
    )

    selected_role = st.selectbox(
        "Select Job Role",
        list(
            ROLE_REQUIREMENTS.keys()
        )
    )

    st.divider()

    st.subheader(
        "Required Skills"
    )

    for skill in ROLE_REQUIREMENTS[
        selected_role
    ]:

        st.write(
            "•",
            skill
        )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Candidate Resume",
    type=[
        "pdf",
        "docx",
        "txt"
    ]
)


if uploaded_file:

    # --------------------------------------------------------
    # Parse document
    # --------------------------------------------------------

    with st.spinner(
        "📖 Reading resume..."
    ):

        try:

            resume_text = extract_document_text(
                uploaded_file
            )

        except Exception as e:

            st.error(
                f"Document parsing failed: {e}"
            )

            st.stop()


    if not resume_text.strip():

        st.error(
            "No text could be extracted from this file."
        )

        st.warning(
            "If this is a scanned/image-only PDF, "
            "OCR will be required."
        )

        st.stop()


    # --------------------------------------------------------
    # Extract skills
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Extracting skills..."
    ):

        candidate_skills = extract_skills(
            resume_text,
            selected_role
        )


    # --------------------------------------------------------
    # Calculate skill match
    # --------------------------------------------------------

    required_skills = ROLE_REQUIREMENTS[
        selected_role
    ]

    (
        matched_skills,
        missing_skills,
        skill_match
    ) = calculate_skill_match(
        required_skills,
        candidate_skills
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model, tokenizer = load_artifacts()

    except Exception as e:

        st.error(
            "Could not load the trained model."
        )

        st.code(
            """
Make sure these files are beside app.py:

resume_match_model.keras
tokenizer.pkl
            """
        )

        st.stop()


    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    with st.spinner(
        "🧠 Running LSTM prediction..."
    ):

        (
            probability,
            prediction,
            model_input
        ) = predict_with_lstm(
            model,
            tokenizer,
            selected_role,
            required_skills,
            candidate_skills
        )


    # ========================================================
    # DASHBOARD
    # ========================================================

    st.success(
        "Resume processed successfully!"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "LSTM Prediction",
            prediction
        )

    with col2:

        st.metric(
            "Suitable Probability",
            f"{probability * 100:.1f}%"
        )

    with col3:

        st.metric(
            "Skill Match",
            f"{skill_match:.1f}%"
        )

    with col4:

        st.metric(
            "Recommendation",
            get_recommendation(
                skill_match
            )
        )


    # ========================================================
    # SKILLS
    # ========================================================

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader(
            "✅ Matched Skills"
        )

        if matched_skills:

            for skill in matched_skills:

                st.success(
                    skill
                )

        else:

            st.warning(
                "No matching skills detected."
            )


    with right:

        st.subheader(
            "❌ Missing Skills"
        )

        if missing_skills:

            for skill in missing_skills:

                st.error(
                    skill
                )

        else:

            st.success(
                "All required skills detected!"
            )


    # ========================================================
    # CANDIDATE SKILLS
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Skills Detected From Resume"
    )

    if candidate_skills:

        st.write(
            ", ".join(
                candidate_skills
            )
        )

    else:

        st.warning(
            "No relevant skills detected."
        )


    # ========================================================
    # RESUME TEXT
    # ========================================================

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.text_area(
            "Resume Text",
            resume_text,
            height=300
        )


    # ========================================================
    # LSTM INPUT
    # ========================================================

    with st.expander(
        "🧠 View LSTM Input"
    ):

        st.code(
            model_input
        )


    # ========================================================
    # REPORT
    # ========================================================

    report = f"""
AI-Powered Resume Screening Report
==================================

Candidate File:
{uploaded_file.name}

Selected Role:
{selected_role}

LSTM Prediction:
{prediction}

Suitable Probability:
{probability * 100:.2f}%

Skill Match:
{skill_match:.2f}%

Recommendation:
{get_recommendation(skill_match)}

Matched Skills:
{", ".join(matched_skills)}

Missing Skills:
{", ".join(missing_skills)}

Detected Candidate Skills:
{", ".join(candidate_skills)}
"""

    st.download_button(
        "⬇️ Download Screening Report",
        report,
        file_name="resume_screening_report.txt",
        mime="text/plain"
    )