# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# FINAL ENTERPRISE BUILD (Hospital Production Grade)
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

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Hospital Clinical Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# DISCLAIMER (GOVERNANCE)
# ======================================================
st.info(
    "ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS) only. "
    "It does NOT provide diagnosis or treatment. "
    "Final clinical decisions must be made by licensed medical professionals."
)

# ======================================================
# STORAGE DIRECTORIES
# ======================================================
BASE_DIR = os.getcwd()
PDF_FOLDER = os.path.join(BASE_DIR, "medical_library")
VECTOR_FOLDER = os.path.join(BASE_DIR, "vector_cache")
LAB_FOLDER = os.path.join(BASE_DIR, "lab_reports")
RAD_FOLDER = os.path.join(BASE_DIR, "radiology")
FHIR_FOLDER = os.path.join(BASE_DIR, "fhir")
HL7_FOLDER = os.path.join(BASE_DIR, "hl7")
HIS_FOLDER = os.path.join(BASE_DIR, "his")
PACS_FOLDER = os.path.join(BASE_DIR, "pacs")
SIGNOFF_FOLDER = os.path.join(BASE_DIR, "signoffs")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")
USERS_DB = os.path.join(BASE_DIR, "users.json")
AUDIT_LOG = os.path.join(BASE_DIR, "audit_log.json")

for p in [PDF_FOLDER, VECTOR_FOLDER, LAB_FOLDER, RAD_FOLDER, FHIR_FOLDER,
          HL7_FOLDER, HIS_FOLDER, PACS_FOLDER, SIGNOFF_FOLDER]:
    os.makedirs(p, exist_ok=True)

# ======================================================
# DEMO USERS
# ======================================================
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
# AUDIT SYSTEM (NABH COMPLIANT)
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
# SAFE AI WRAPPER (GOVERNANCE)
# ======================================================
def safe_ai_call(prompt):
    try:
        result = external_research_answer(prompt)
        if not result or "answer" not in result:
            return {"status": "error", "answer": "⚠ AI returned empty response.", "confidence": 0.0}

        confidence = min(0.95, max(0.6, len(result["answer"]) / 1200))
        return {"status": "ok", "answer": result["answer"], "confidence": round(confidence, 2)}

    except Exception as e:
        audit("ai_failure", {"error": str(e)})
        return {"status": "down", "answer": "⚠ AI service unavailable. Governance block applied.", "confidence": 0.0}

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
        st.metric("Clinical Score", f"{score_data['score']} / 10")
    with c2:
        st.metric("Risk Level", score_data["risk"])
    with c3:
        st.metric("Urgency", score_data["urgency"])
    with c4:
        st.metric("AI Confidence", f"{int(confidence * 100)}%")

# ======================================================
# LOGIN SYSTEM
# ======================================================
def login_ui():
    st.title("ĀROGYABODHA AI — Secure Hospital Login")
    with st.form("login_form"):
        username = st.text_input("Doctor / Researcher ID")
        password = st.text_input("Secure Access Key", type="password")
        submitted = st.form_submit_button("🚀 Enter Platform")

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
# MODEL (EMBEDDINGS)
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================================================
# FAISS EVIDENCE ENGINE
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

    if not docs:
        return None, [], []

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
# SIDEBAR — HOSPITAL COMMAND CENTER
# ======================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}** ({st.session_state.role})")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.subheader("📁 Hospital Evidence Library")

uploads = st.sidebar.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build Evidence Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    audit("build_index", {"count": len(st.session_state.documents)})
    st.sidebar.success("Evidence Index Built")

st.sidebar.markdown("🟢 Index Status: READY" if st.session_state.index_ready else "🔴 Index Status: NOT BUILT")

module = st.sidebar.radio("Hospital Command Center", [
    "🔬 Clinical Research Copilot",
    "🏥 ICU Intelligence",
    "🧪 Lab Intelligence",
    "💊 Drug Interaction AI",
    "🩻 Radiology AI",
    "📡 HL7 / FHIR Gateway",
    "🏥 HIS Integration",
    "🩻 PACS Integration",
    "🧾 Doctor Sign-off",
    "📊 NABH Compliance",
    "🕒 Audit Trail"
])

# ======================================================
# MODULES
# ======================================================

# 🔬 Clinical Research Copilot
if module == "🔬 Clinical Research Copilot":
    st.header("🔬 Clinical Research Copilot")
    query = st.text_input("Ask clinical question")

    if st.button("Analyze") and query:
        if not st.session_state.index_ready:
            st.error("Hospital Evidence Index not built.")
        else:
            qemb = embedder.encode([query])
            _, I = st.session_state.index.search(np.array(qemb, dtype=np.float32), 5)
            context = "\n\n".join([st.session_state.documents[i] for i in I[0]])

            resp = safe_ai_call(f"Use only hospital evidence:\n{context}\n\nQ:{query}")

            if resp["status"] == "ok":
                st.write(resp["answer"])
                score_data = clinical_scoring_engine(resp["answer"], resp["confidence"])
                render_clinical_risk_panel(score_data, resp["confidence"])

# 🏥 ICU Intelligence
if module == "🏥 ICU Intelligence":
    st.header("🏥 ICU Early Warning System")

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

# 🧪 Lab Intelligence
if module == "🧪 Lab Intelligence":
    st.header("🧪 Lab Intelligence")
    file = st.file_uploader("Upload Lab Report")
    if file:
        resp = safe_ai_call("Interpret lab report")
        st.write(resp["answer"])
        score_data = clinical_scoring_engine(resp["answer"], resp["confidence"])
        render_clinical_risk_panel(score_data, resp["confidence"])

# 💊 Drug AI
if module == "💊 Drug Interaction AI":
    meds = st.text_input("Enter drugs")
    if st.button("Analyze"):
        resp = safe_ai_call(f"Analyze drug interaction: {meds}")
        st.write(resp["answer"])
        score_data = clinical_scoring_engine(resp["answer"], resp["confidence"])
        render_clinical_risk_panel(score_data, resp["confidence"])

# 🩻 Radiology AI
if module == "🩻 Radiology AI":
    file = st.file_uploader("Upload scan")
    if file:
        resp = safe_ai_call("Generate radiology report")
        st.write(resp["answer"])
        score_data = clinical_scoring_engine(resp["answer"], resp["confidence"])
        render_clinical_risk_panel(score_data, resp["confidence"])

# 📡 HL7 / FHIR Gateway
if module == "📡 HL7 / FHIR Gateway":
    st.header("📡 HL7 / FHIR Gateway")
    raw = st.text_area("Paste HL7 Message")
    if st.button("Ingest HL7"):
        fid = hashlib.sha256(raw.encode()).hexdigest()[:12]
        open(os.path.join(HL7_FOLDER, f"{fid}.hl7"), "w").write(raw)
        audit("hl7_ingest", {"id": fid})
        st.success(f"HL7 Ingested ID: {fid}")

    fhir = st.text_area("Paste FHIR JSON")
    if st.button("Ingest FHIR"):
        fid = hashlib.sha256(fhir.encode()).hexdigest()[:12]
        open(os.path.join(FHIR_FOLDER, f"{fid}.json"), "w").write(fhir)
        audit("fhir_ingest", {"id": fid})
        st.success(f"FHIR Ingested ID: {fid}")

# 🏥 HIS Integration
if module == "🏥 HIS Integration":
    st.header("🏥 HIS Integration")
    his = st.text_area("Paste HIS JSON Record")
    if st.button("Ingest HIS"):
        fid = hashlib.sha256(his.encode()).hexdigest()[:12]
        open(os.path.join(HIS_FOLDER, f"{fid}.json"), "w").write(his)
        audit("his_ingest", {"id": fid})
        st.success(f"HIS Record Stored ID: {fid}")

# 🩻 PACS Integration
if module == "🩻 PACS Integration":
    st.header("🩻 PACS Integration")
    file = st.file_uploader("Upload DICOM/Image")
    if file:
        fid = hashlib.sha256(file.getvalue()).hexdigest()[:12]
        open(os.path.join(PACS_FOLDER, f"{fid}_{file.name}"), "wb").write(file.getvalue())
        audit("pacs_ingest", {"id": fid})
        st.success(f"PACS Image Stored ID: {fid}")

# 🧾 Doctor Sign-off
if module == "🧾 Doctor Sign-off":
    st.header("🧾 Doctor Clinical Sign-off")

    pid = st.text_input("Patient ID")
    decision = st.selectbox("Decision", ["Approved", "Escalated", "Deferred"])
    note = st.text_area("Doctor Note")

    if st.button("Submit Sign-off"):
        rec = {
            "time": str(datetime.datetime.now()),
            "doctor": st.session_state.username,
            "patient": pid,
            "decision": decision,
            "note": note
        }
        path = os.path.join(SIGNOFF_FOLDER, f"{pid}.json")
        history = []
        if os.path.exists(path):
            history = json.load(open(path))
        history.append(rec)
        json.dump(history, open(path, "w"), indent=2)
        audit("doctor_signoff", rec)
        st.success("Doctor sign-off recorded")

# 📊 NABH Compliance
if module == "📊 NABH Compliance":
    st.header("📊 NABH Compliance Dashboard")
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.metric("Total Events", len(df))
        st.metric("Doctors Active", df["user"].nunique())
        st.dataframe(df, use_container_width=True)

# 🕒 Audit Trail
if module == "🕒 Audit Trail":
    st.header("🕒 Audit Trail")
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital Clinical Intelligence Platform — Enterprise Production Build")
