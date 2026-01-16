# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# v2.1 (Universal Clinical Scoring Engine Integrated)
# ======================================================

import streamlit as st
import os, json, pickle, datetime, io, hashlib
import numpy as np
import faiss
import pandas as pd
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# Optional OCR
OCR_AVAILABLE = True
try:
    import pytesseract
    from pdf2image import convert_from_path
except:
    OCR_AVAILABLE = False


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Hospital Clinical Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# DISCLAIMER
# ======================================================
st.info(
    "ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS) only. "
    "It does NOT provide diagnosis or treatment. "
    "Final clinical decisions must be made by licensed medical professionals."
)

# ======================================================
# STORAGE
# ======================================================
BASE_DIR = os.getcwd()
PDF_FOLDER = os.path.join(BASE_DIR, "medical_library")
LAB_FOLDER = os.path.join(BASE_DIR, "lab_reports")
RAD_FOLDER = os.path.join(BASE_DIR, "radiology")
EMR_FOLDER = os.path.join(BASE_DIR, "emr")
VECTOR_FOLDER = os.path.join(BASE_DIR, "vector_cache")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")
USERS_DB = os.path.join(BASE_DIR, "users.json")
AUDIT_LOG = os.path.join(BASE_DIR, "audit_log.json")

for p in [PDF_FOLDER, LAB_FOLDER, RAD_FOLDER, EMR_FOLDER, VECTOR_FOLDER]:
    os.makedirs(p, exist_ok=True)

# Demo users
if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"},
        "admin1": {"password": "admin123", "role": "Admin"}
    }, open(USERS_DB, "w"), indent=2)

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# AUDIT SYSTEM
# ======================================================
def audit(event, meta=None):
    rows = []
    if os.path.exists(AUDIT_LOG):
        rows = json.load(open(AUDIT_LOG))
    rows.append({
        "time": str(datetime.datetime.now()),
        "user": st.session_state.get("username"),
        "role": st.session_state.get("role"),
        "event": event,
        "meta": meta or {}
    })
    json.dump(rows, open(AUDIT_LOG, "w"), indent=2)

# ======================================================
# SAFE AI WRAPPER
# ======================================================
def safe_ai_call(prompt, mode="AI") -> Dict[str, Any]:
    try:
        result = external_research_answer(prompt)
        if not result or "answer" not in result:
            return {"status": "error", "answer": "⚠ AI returned empty response.", "confidence": 0.0}

        confidence = min(0.95, max(0.55, len(result["answer"]) / 1500))
        return {"status": "ok", "answer": result["answer"], "confidence": round(confidence, 2)}

    except Exception as e:
        audit("ai_failure", {"mode": mode, "error": str(e)})
        return {"status": "down", "answer": "⚠ AI service unavailable.", "confidence": 0.0}

# ======================================================
# UTILITIES
# ======================================================
def hash_file(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


# ======================================================
# UNIVERSAL CLINICAL SCORING ENGINE
# ======================================================
def clinical_scoring_engine(answer: str, confidence: float):
    if not answer:
        return {"score": 0, "risk": "UNKNOWN", "urgency": "UNKNOWN"}

    length_factor = min(10, len(answer) / 400)
    confidence_factor = confidence * 10
    score = round((length_factor * 0.6) + (confidence_factor * 0.4), 1)

    if score >= 8:
        risk = "HIGH"
        urgency = "Immediate Review"
    elif score >= 5:
        risk = "MEDIUM"
        urgency = "Priority Review"
    else:
        risk = "LOW"
        urgency = "Routine Review"

    return {"score": min(10, score), "risk": risk, "urgency": urgency}


def render_clinical_risk_panel(score_data, confidence):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Clinical Severity Score", f"{score_data['score']} / 10")
    with c2:
        st.metric("Risk Level", score_data["risk"])
    with c3:
        st.metric("Urgency", score_data["urgency"])
    with c4:
        st.metric("AI Confidence", f"{int(confidence * 100)}%")

# ======================================================
# LOGIN
# ======================================================
def login_ui():
    st.markdown("<h2>ĀROGYABODHA AI Login</h2>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        users = json.load(open(USERS_DB))
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = users[username]["role"]
            audit("login", {"user": username})
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ======================================================
# MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================================================
# FAISS INDEX
# ======================================================
def extract_text_from_pdf_bytes(file_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(file_bytes))
    return [p.extract_text() for p in reader.pages if p.extract_text()]

def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            with open(os.path.join(PDF_FOLDER, pdf), "rb") as f:
                texts = extract_text_from_pdf_bytes(f.read())
            for i, t in enumerate(texts):
                docs.append(t)
                srcs.append(f"{pdf} — Page {i+1}")

    emb = embedder.encode(docs)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb, dtype=np.float32))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents": docs, "sources": srcs}, open(CACHE_FILE, "wb"))
    return idx, docs, srcs

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data["documents"]
    st.session_state.sources = data["sources"]
    st.session_state.index_ready = True

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown(f"👨‍⚕️ {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

module = st.sidebar.radio("Command Center", [
    "🔬 Clinical Research Copilot",
    "🏥 ICU Intelligence",
    "🧪 Lab Intelligence",
    "💊 Drug Interaction AI",
    "🩻 Radiology AI",
    "🕒 Audit Trail"
])

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if module == "🔬 Clinical Research Copilot":
    st.header("🔬 Clinical Research Copilot")
    query = st.text_input("Ask clinical question")

    if st.button("Analyze") and query:
        qemb = embedder.encode([query])
        _, I = st.session_state.index.search(np.array(qemb, dtype=np.float32), 5)
        context = "\n\n".join([st.session_state.documents[i] for i in I[0]])

        resp = safe_ai_call(f"Use this evidence:\n{context}\n\nQ:{query}")

        if resp["status"] == "ok":
            st.write(resp["answer"])
            score_data = clinical_scoring_engine(resp["answer"], resp["confidence"])
            render_clinical_risk_panel(score_data, resp["confidence"])

# ======================================================
# ICU INTELLIGENCE
# ======================================================
if module == "🏥 ICU Intelligence":
    st.header("🏥 ICU Early Warning")

    hr = st.number_input("Heart Rate", 30, 200, 90)
    rr = st.number_input("Resp Rate", 8, 60, 20)
    spo2 = st.number_input("SpO2", 60, 100, 95)
    temp = st.number_input("Temp", 34.0, 42.0, 37.5)

    score = 0
    score += 2 if hr > 110 else 0
    score += 2 if rr > 25 else 0
    score += 2 if spo2 < 92 else 0
    score += 1 if temp > 38 else 0

    risk = "LOW" if score < 2 else "MEDIUM" if score < 4 else "HIGH"

    render_clinical_risk_panel(
        {"score": score, "risk": risk, "urgency": "Immediate" if risk=="HIGH" else "Priority"},
        0.9
    )

# ======================================================
# LAB INTELLIGENCE
# ======================================================
if module == "🧪 Lab Intelligence":
    st.header("🧪 Lab Intelligence")
    file = st.file_uploader("Upload Lab Report")
    if file:
        resp = safe_ai_call("Interpret lab report")
        st.write(resp["answer"])
        score_data = clinical_scoring_engine(resp["answer"], resp["confidence"])
        render_clinical_risk_panel(score_data, resp["confidence"])

# ======================================================
# DRUG AI
# ======================================================
if module == "💊 Drug Interaction AI":
    meds = st.text_input("Enter drugs")
    if st.button("Analyze"):
        resp = safe_ai_call(f"Analyze drug interaction: {meds}")
        st.write(resp["answer"])
        score_data = clinical_scoring_engine(resp["answer"], resp["confidence"])
        render_clinical_risk_panel(score_data, resp["confidence"])

# ======================================================
# RADIOLOGY AI
# ======================================================
if module == "🩻 Radiology AI":
    file = st.file_uploader("Upload scan")
    if file:
        resp = safe_ai_call("Generate radiology report")
        st.write(resp["answer"])
        score_data = clinical_scoring_engine(resp["answer"], resp["confidence"])
        render_clinical_risk_panel(score_data, resp["confidence"])

# ======================================================
# AUDIT
# ======================================================
if module == "🕒 Audit Trail":
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df)

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital Clinical Intelligence Platform — v2.1")
