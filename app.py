import streamlit as st
import pickle
import re
from PyPDF2 import PdfReader

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Resume Analyzer", layout="wide")

st.title("📄 Resume Analyzer & Job Recommendation System")

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_models():
    rf_cat = pickle.load(open("models/rf_classifier_categorization.pkl", "rb"))
    tfidf_cat = pickle.load(open("models/tfidf_vectorizer_categorization.pkl", "rb"))
    rf_job = pickle.load(open("models/rf_classifier_job_recommendation.pkl", "rb"))
    tfidf_job = pickle.load(open("models/tfidf_vectorizer_job_recommendation.pkl", "rb"))
    return rf_cat, tfidf_cat, rf_job, tfidf_job

rf_classifier_categorization, tfidf_vectorizer_categorization, \
rf_classifier_job_recommendation, tfidf_vectorizer_job_recommendation = load_models()

# ===================== UTIL FUNCTIONS =====================
def cleanResume(txt):
    txt = re.sub(r'http\S+', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def predict_category(text):
    text = cleanResume(text)
    tfidf = tfidf_vectorizer_categorization.transform([text])
    return rf_classifier_categorization.predict(tfidf)[0]

def job_recommendation(text):
    text = cleanResume(text)
    tfidf = tfidf_vectorizer_job_recommendation.transform([text])
    return rf_classifier_job_recommendation.predict(tfidf)[0]

# ===================== EXTRACTION FUNCTIONS =====================
def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else "Not Found"

def extract_email_from_resume(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else "Not Found"

def extract_name_from_resume(text):
    pattern = r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b"
    match = re.search(pattern, text)
    return match.group() if match else "Not Found"

def extract_skills_from_resume(text):
    skills_list = [
        "Python","Machine Learning","Data Analysis","SQL","Java","C++","JavaScript",
        "HTML","CSS","React","Django","Flask","FastAPI","TensorFlow","Keras",
        "PyTorch","NLP","Scikit-learn","Pandas","NumPy","Power BI","Tableau"
    ]

    found_skills = []
    for skill in skills_list:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            found_skills.append(skill)
    return found_skills if found_skills else ["Not Found"]

def extract_education_from_resume(text):
    education_keywords = [
        "Computer Science","Information Technology","Data Science","Engineering",
        "Bachelor","Master","B.Tech","M.Tech","B.Sc","M.Sc","PhD"
    ]

    found_edu = []
    for edu in education_keywords:
        if re.search(rf"\b{re.escape(edu)}\b", text, re.IGNORECASE):
            found_edu.append(edu)
    return found_edu if found_edu else ["Not Found"]

# ===================== UI =====================
uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        resume_text = pdf_to_text(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")

    if st.button("🔍 Analyze Resume"):
        with st.spinner("Analyzing resume..."):
            category = predict_category(resume_text)
            job = job_recommendation(resume_text)
            phone = extract_contact_number_from_resume(resume_text)
            email = extract_email_from_resume(resume_text)
            name = extract_name_from_resume(resume_text)
            skills = extract_skills_from_resume(resume_text)
            education = extract_education_from_resume(resume_text)

        st.success("Analysis Complete ✅")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 Candidate Details")
            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
            st.write(f"**Phone:** {phone}")

        with col2:
            st.subheader("🎯 Predictions")
            st.write(f"**Resume Category:** {category}")
            st.write(f"**Recommended Job:** {job}")

        st.subheader("🛠 Skills")
        st.write(", ".join(skills))

        st.subheader("🎓 Education")
        st.write(", ".join(education))
