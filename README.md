# AI-Powered Resume Screening & Candidate Ranking System

An AI-powered resume screening application that uses **Natural Language Processing (NLP), Deep Learning (LSTM), and automated skill matching** to evaluate candidate suitability for different job roles.

The system extracts text from uploaded resumes, identifies relevant technical skills, compares them against job requirements, predicts candidate suitability using an LSTM model, and generates an explainable screening report.

## 🚀 Project Overview

Recruiters often receive hundreds of resumes for a single job opening. Manually reviewing every resume is time-consuming and makes initial candidate screening difficult.

This project aims to simplify the initial screening process by automating resume parsing, skill extraction, and candidate-job compatibility assessment.

### Key Objectives

- Automate resume text extraction.
- Identify relevant skills from candidate resumes.
- Match candidate skills against job-specific requirements.
- Predict candidate suitability using an LSTM model.
- Identify matched and missing skills.
- Generate candidate screening recommendations.

## ✨ Features

### 1. Real-Time Resume Parsing
- Upload resumes in PDF, DOCX, and TXT formats.
- Extract text automatically from uploaded documents.
- Display extracted resume content.

### 2. Automated Skill Extraction
- Identify relevant technical skills from resume content.
- Support different job-specific skill requirements.
- Handle common skill name variations and aliases.

### 3. LSTM-Based Suitability Prediction
- Uses a trained Long Short-Term Memory (LSTM) neural network.
- Processes job role, required skills, and candidate skills.
- Predicts:
  - Suitable
  - Not Suitable
- Displays the model's suitability probability.

### 4. Skill Matching & Analysis
- Calculate candidate skill-match percentage.
- Identify matched skills.
- Identify missing skills.
- Generate suitability recommendations.

### 5. Interactive Streamlit Dashboard
- Select different job roles.
- Upload and analyze resumes.
- View screening results.
- Download screening reports.

## 🧠 Supported Job Roles

The current implementation supports four job roles:

| Job Role | Example Required Skills |
|---|---|
| Data Analyst | Python, SQL, Excel, Power BI, Tableau, Pandas |
| Machine Learning Engineer | Python, TensorFlow, PyTorch, Scikit-learn, Deep Learning |
| Full Stack Developer | HTML, CSS, JavaScript, React, Node.js, MongoDB |
| Business Analyst | Excel, SQL, Power BI, Statistics, Agile, JIRA |

## ⚙️ Machine Learning Architecture

The LSTM model follows this architecture:

**Embedding → LSTM → Dropout → Dense → Output**

### NLP Processing Pipeline

1. Text preprocessing
2. Tokenization
3. Vocabulary building
4. Sequence conversion
5. Padding
6. Embedding representation
7. LSTM-based classification

### Model Input

The model receives a combined text representation containing:

- Job Role
- Required Skills
- Candidate Skills

### Model Output

Binary classification:

- `1` – Suitable
- `0` – Not Suitable

The model also provides a prediction probability.

## 📊 Dataset

The LSTM model was trained using a **synthetic dataset containing 6,000 records**.

The dataset contains skills-based candidate and job requirement combinations across four job roles.

The training data was designed to provide examples of different levels of skill compatibility.

**Note:** The dataset is synthetic. The reported model performance should not be interpreted as real-world hiring accuracy or evidence of recruiter-level assessment.

## 📈 Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 86% |
| Test Loss | 0.3214 |

The model was evaluated using a held-out test dataset.

## 🏗️ Project Architecture

```text
AI_RESUME_SCREENER/
│
├── backend/
│   ├── __init__.py
│   ├── document_parser.py
│   ├── skill_extractor.py
│   ├── model_service.py
│   └── screening.py
│
├── frontend/
│   └── app.py
│
├── notebooks/
│   ├── resume_match_model.keras
│   ├── tokenizer.pkl
│   └── lstm_dataset_harder_v2.csv
│
├── dataset/
│
├── requirements.txt
│
└── README.md
```

### Backend Modules

| Module | Responsibility |
|---|---|
| document_parser.py | Extract text from PDF, DOCX, and TXT files |
| skill_extractor.py | Skill extraction and skill matching |
| model_service.py | Load trained LSTM model and tokenizer; perform predictions |
| screening.py | Combine screening results and generate recommendations |

### Frontend

`app.py` handles the Streamlit interface, file uploads, job-role selection, and result visualization.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| NLP | Tokenization, Sequence Padding, Embedding |
| Data Processing | Pandas, NumPy |
| Document Parsing | PyPDF, python-docx |
| Frontend | Streamlit |
| Model Storage | Keras `.keras`, Pickle |

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Resume-Screener.git
```

### 2. Navigate to the Project Folder

```bash
cd AI-Resume-Screener
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

From the project root directory:

```bash
python -m streamlit run frontend/app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 📄 How to Use

1. Launch the Streamlit application.
2. Select the target job role.
3. Upload a candidate resume in PDF, DOCX, or TXT format.
4. Wait for automatic document parsing.
5. Review the extracted candidate skills.
6. View the LSTM suitability prediction and probability.
7. Analyze matched and missing skills.
8. Review the final suitability recommendation.
9. Download the screening report.

## 📌 Recommendation Logic

The application generates recommendations based on skill-match percentage:

| Skill Match | Recommendation |
|---|---|
| 80% – 100% | Highly Suitable |
| 50% – 79.99% | Moderately Suitable |
| Below 50% | Not Suitable |

The recommendation is generated by the separate rule-based skill-matching component, while the LSTM produces its own binary suitability prediction.

These are separate outputs and may differ.

## 🔮 Future Enhancements

- OCR support for scanned PDF resumes.
- Advanced NLP-based skill extraction.
- Custom job description uploads.
- Candidate ranking and comparison dashboard.
- Multi-resume batch processing.
- Improved resume parsing accuracy.
- Deployment using Streamlit Community Cloud or another hosting platform.
- More diverse real-world resume datasets for model evaluation.

## ⚠️ Limitations

- The current skill extraction system uses predefined skill dictionaries and aliases.
- The current LSTM model is trained on synthetic data.
- The system evaluates skills-based compatibility, not complete candidate suitability.
- Scanned PDFs require OCR support.
- The model's prediction probability is not a calibrated hiring probability.

This project is intended as an educational and decision-support prototype. It should not be used as the sole basis for real hiring decisions.

## 👨‍💻 Author

**Syed Abdul Kathir**

B.Tech – Artificial Intelligence and Data Science

Interested in Artificial Intelligence, Machine Learning, Deep Learning, NLP, and Generative AI.

---

⭐ If you find this project interesting, consider giving the repository a star!
