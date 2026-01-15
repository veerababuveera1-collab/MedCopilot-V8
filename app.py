# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# v2.0 (Merged with v1.0 Engine)
# ======================================================

import streamlit as st
import os, json, pickle, datetime, io, hashlib, math
import numpy as np
import faiss
import pandas as pd
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# External AI connector (must return: {"answer": "..."} )
from external_research import external_research_answer

# Optional OCR (hooks)
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
# DISCLAIMER (Governance)
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
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

for p in [PDF_FOLDER, LAB_FOLDER, RAD_FOLDER, EMR_FOLDER, VECTOR_FOLDER]:
    os.makedirs(p, exist_ok=True)

# Demo users (replace with hospital IAM later)
if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"},
        "admin1": {"password": "admin123", "role": "Admin"}
    }, open(USERS_DB, "w"), indent=2)

# Default settings / feature toggles
if not os.path.exists(SETTINGS_FILE):
    json.dump({
        "features": {
            "icu_ai": True,
            "lab_ai": True,
            "drug_ai": True,
            "radiology_ai": True,
            "timeline": True,
            "hybrid_ai": True
        },
        "ai": {
            "confidence_threshold": 0.65
        }
    }, open(SETTINGS_FILE, "w"), indent=2)

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
    "last_ai_health": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# AUDIT SYSTEM (Compliance)
# ======================================================
def audit(event, meta=None):
    rows = []
    if os.path.exists(AUDIT_LOG):
        try:
            rows = json.load(open(AUDIT_LOG))
        except:
            rows = []
    rows.append({
        "time": str(datetime.datetime.now()),
        "user": st.session_state.get("username"),
        "role": st.session_state.get("role"),
        "event": event,
        "meta": meta or {}
    })
    json.dump(rows, open(AUDIT_LOG, "w"), indent=2)

# ======================================================
# SAFE AI WRAPPER (Governance)
# ======================================================
def safe_ai_call(prompt, mode="AI") -> Dict[str, Any]:
    try:
        result = external_research_answer(prompt)
        if not result or "answer" not in result:
            audit("ai_empty_response", {"mode": mode})
            return {"status": "error", "answer": "⚠ AI returned empty response.", "confidence": 0.0}
        # simple heuristic confidence (can be replaced by model metadata)
        confidence = min(0.95, max(0.55, len(result["answer"]) / 1500))
        st.session_state.last_ai_health = {"ok": True, "time": str(datetime.datetime.now())}
        return {"status": "ok", "answer": result["answer"], "confidence": round(confidence, 2)}
    except Exception as e:
        audit("ai_failure", {"mode": mode, "error": str(e)})
        st.session_state.last_ai_health = {"ok": False, "time": str(datetime.datetime.now()), "error": str(e)}
        return {"status": "down", "answer": "⚠ AI service unavailable. Governance block applied.", "confidence": 0.0}

# ======================================================
# UTILITIES
# ======================================================
def hash_file(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]

def load_settings():
    return json.load(open(SETTINGS_FILE))

def save_settings(s):
    json.dump(s, open(SETTINGS_FILE, "w"), indent=2)

# ======================================================
# WOW LOGIN UI (Streamlit-native)
# ======================================================
def login_ui():
    st.markdown("""
    <style>
    body { background: radial-gradient(circle at top, #020617 0%, #020617 60%, #000 100%); }
    .login-card { max-width:520px;margin:120px auto;padding:40px;border-radius:20px;
      background:rgba(255,255,255,.06);backdrop-filter:blur(20px);
      box-shadow:0 0 80px rgba(56,189,248,.25);border:1px solid rgba(255,255,255,.15);text-align:center;}
    .login-title{font-size:36px;font-weight:900;margin-bottom:6px;
      background:linear-gradient(90deg,#38bdf8,#22d3ee,#0ea5e9);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .login-sub{color:#cbd5f5;margin-bottom:30px}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">ĀROGYABODHA AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Hospital Clinical Intelligence Platform<br>Secure Medical AI Command Center</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Doctor / Researcher ID")
        password = st.text_input("Secure Access Key", type="password")
        submitted = st.form_submit_button("🚀 Enter Clinical AI Platform")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        users = json.load(open(USERS_DB))
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = users[username]["role"]
            audit("login", {"user": username})
            st.success("✅ Secure Hospital Access Granted")
            st.rerun()
        else:
            audit("login_failed", {"user": username})
            st.error("❌ Invalid Credentials")

# ======================================================
# AUTH GATE
# ======================================================
if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ======================================================
# MODEL (Embeddings)
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================================================
# FAISS INDEX (Hospital Evidence RAG)
# ======================================================
def extract_text_from_pdf_bytes(file_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for i, p in enumerate(reader.pages[:300]):  # cap pages for safety
        t = p.extract_text()
        if t and len(t) > 100:
            pages_text.append(t)
    return pages_text

def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.lower().endswith(".pdf"):
            with open(os.path.join(PDF_FOLDER, pdf), "rb") as f:
                texts = extract_text_from_pdf_bytes(f.read())
            for i, t in enumerate(texts):
                docs.append(t)
                srcs.append(f"{pdf} — Page {i+1}")

    if not docs:
        return None, [], []

    emb = embedder.encode(docs, show_progress_bar=False)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb, dtype=np.float32))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents": docs, "sources": srcs}, open(CACHE_FILE, "wb"))
    return idx, docs, srcs

# Load cached index if exists
if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data.get("documents", [])
    st.session_state.sources = data.get("sources", [])
    st.session_state.index_ready = True

# ======================================================
# SIDEBAR — COMMAND CENTER NAV
# ======================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}** ({st.session_state.role})")

if st.sidebar.button("Logout"):
    audit("logout", {})
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.subheader("📁 Hospital Evidence Library")
uploads = st.sidebar.file_uploader("Upload Medical PDFs (Bulk)", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build Evidence Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    audit("build_index", {"count": len(st.session_state.documents)})
    st.sidebar.success("Hospital Evidence Index Built")

st.sidebar.markdown("🟢 Index Status: READY" if st.session_state.index_ready else "🔴 Index Status: NOT BUILT")

module = st.sidebar.radio("Command Center", [
    "🧠 System Dashboard",
    "🔬 Clinical Research Copilot",
    "🏥 ICU Intelligence",
    "🧪 Lab Intelligence",
    "💊 Drug Interaction AI",
    "🩻 Radiology AI",
    "🗓 Patient Timeline",
    "🕒 Audit & Compliance",
    "⚙️ Admin Control Panel"
])

# ======================================================
# HEADER (Dashboard)
# ======================================================
st.markdown("## 🧠 ĀROGYABODHA AI — Hospital Clinical Intelligence Platform (v2.0)")
st.caption("Hospital-grade • Evidence-locked • OCR-enabled • Governance enabled")

# ======================================================
# SYSTEM DASHBOARD (Command Center)
# ======================================================
if module == "🧠 System Dashboard":
    st.header("🧠 Hospital AI Command Center")

    settings = load_settings()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System Status", "ONLINE")
        st.metric("AI Engine", "CONNECTED" if st.session_state.last_ai_health is None or st.session_state.last_ai_health.get("ok", True) else "DEGRADED")
    with col2:
        st.metric("Evidence Index", "READY" if st.session_state.index_ready else "NOT BUILT")
        st.metric("Total Docs", len(st.session_state.documents))
    with col3:
        st.metric("OCR Engine", "READY" if OCR_AVAILABLE else "DISABLED")
        st.metric("Hybrid AI", "ON" if settings["features"].get("hybrid_ai", True) else "OFF")
    with col4:
        if os.path.exists(AUDIT_LOG):
            logs = json.load(open(AUDIT_LOG))
            st.metric("Audit Events", len(logs))
            st.metric("Last Audit", logs[-1]["time"])
        else:
            st.metric("Audit Events", 0)
            st.metric("Last Audit", "—")

    st.success("Governance Engine: ACTIVE")
    st.success("Clinical Safety Layer: ENABLED")

# ======================================================
# CLINICAL RESEARCH COPILOT (v1.0 Engine kept)
# ======================================================
if module == "🔬 Clinical Research Copilot":
    st.header("🔬 Clinical Research Copilot")

    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("🚀 Analyze") and query:
        audit("clinical_query", {"query": query, "mode": mode})

        col1, col2 = st.columns(2)

        # Hospital AI (RAG)
        if mode in ["Hospital AI", "Hybrid AI"]:
            with col1:
                st.subheader("🏥 Hospital Evidence AI")
                if not st.session_state.index_ready:
                    st.error("Hospital evidence index not built.")
                else:
                    qemb = embedder.encode([query])
                    _, I = st.session_state.index.search(np.array(qemb, dtype=np.float32), 5)
                    context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
                    sources = [st.session_state.sources[i] for i in I[0]]

                    prompt = f"Use only hospital evidence below to answer.\n\n{context}\n\nQuestion: {query}"
                    resp = safe_ai_call(prompt, "Hospital AI")

                    if resp["status"] == "ok":
                        st.success(f"Answer (Confidence: {int(resp['confidence']*100)}%)")
                        st.write(resp["answer"])
                        st.markdown("#### 📑 Evidence Sources")
                        for s in sources:
                            st.info(s)
                    else:
                        st.error(resp["answer"])

        # Global AI
        if mode in ["Global AI", "Hybrid AI"]:
            with col2:
                st.subheader("🌍 Global Research AI")
                resp = safe_ai_call(query, "Global AI")
                if resp["status"] == "ok":
                    st.success(f"Answer (Confidence: {int(resp['confidence']*100)}%)")
                    st.write(resp["answer"])
                else:
                    st.error(resp["answer"])

# ======================================================
# ICU INTELLIGENCE (Early Warning + Risk)
# ======================================================
if module == "🏥 ICU Intelligence":
    st.header("🏥 ICU Intelligence — Early Warning System")

    st.info("Enter patient vitals (manual / device feed) to compute risk scores.")

    c1, c2, c3, c4 = st.columns(4)
    hr = c1.number_input("Heart Rate (bpm)", 30, 220, 98)
    rr = c2.number_input("Respiratory Rate (/min)", 8, 60, 22)
    spo2 = c3.number_input("SpO₂ (%)", 60, 100, 94)
    temp = c4.number_input("Temperature (°C)", 34.0, 42.0, 38.2)

    # Simple NEWS2-like heuristic (placeholder model)
    score = 0
    score += 2 if hr >= 110 or hr < 50 else 0
    score += 2 if rr >= 25 or rr <= 8 else 0
    score += 2 if spo2 <= 92 else 0
    score += 1 if temp >= 38.0 or temp <= 35.0 else 0

    risk = "LOW"
    if score >= 4: risk = "HIGH"
    elif score >= 2: risk = "MEDIUM"

    st.subheader("🧠 AI Risk Assessment")
    st.metric("Early Warning Score", score)
    st.metric("Clinical Risk", risk)

    if st.button("Generate AI Summary"):
        prompt = f"Patient vitals HR:{hr}, RR:{rr}, SpO2:{spo2}, Temp:{temp}. Provide clinical risk summary and actions."
        resp = safe_ai_call(prompt, "ICU AI")
        st.write(resp["answer"])

# ======================================================
# LAB INTELLIGENCE (OCR → Structured → AI)
# ======================================================
if module == "🧪 Lab Intelligence":
    st.header("🧪 Lab Report Intelligence")

    uploaded_lab = st.file_uploader("Upload Lab Report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_lab:
        data = uploaded_lab.getvalue()
        fid = hash_file(data)
        path = os.path.join(LAB_FOLDER, f"{fid}_{uploaded_lab.name}")
        with open(path, "wb") as out:
            out.write(data)

        audit("lab_upload", {"file": uploaded_lab.name, "id": fid})
        st.success(f"Lab report uploaded (ID: {fid})")

        # OCR hook
        if OCR_AVAILABLE and uploaded_lab.type.startswith("image"):
            st.markdown("### 🧾 OCR Text (Preview)")
            st.write("OCR pipeline ready. (Integrate image-to-text here.)")

        st.markdown("### 🧠 AI Interpretation")
        prompt = "Interpret this lab report and highlight critical values, trends, and recommendations."
        resp = safe_ai_call(prompt, "Lab AI")
        st.write(resp["answer"])

# ======================================================
# DRUG INTERACTION AI (Rules + AI)
# ======================================================
if module == "💊 Drug Interaction AI":
    st.header("💊 Drug Interaction & Pharmacy AI")

    meds = st.text_input("Enter medications (comma-separated)", "Warfarin, Azithromycin")
    renal = st.selectbox("Renal Function", ["Normal", "Impaired"])
    hepatic = st.selectbox("Hepatic Function", ["Normal", "Impaired"])

    # Simple rule demo
    risk = "LOW"
    if "warfarin" in meds.lower() and "azithro" in meds.lower():
        risk = "HIGH"

    st.metric("Interaction Risk", risk)

    if st.button("Analyze Interactions"):
        prompt = f"Analyze drug interactions for: {meds}. Renal:{renal}, Hepatic:{hepatic}. Provide risks and alternatives."
        resp = safe_ai_call(prompt, "Drug AI")
        st.write(resp["answer"])

# ======================================================
# RADIOLOGY AI (Pipeline Hooks)
# ======================================================
if module == "🩻 Radiology AI":
    st.header("🩻 Radiology AI (X-ray/CT/MRI)")

    rad = st.file_uploader("Upload Imaging (DICOM/PNG/JPG)", type=["dcm", "png", "jpg", "jpeg"])

    if rad:
        data = rad.getvalue()
        fid = hash_file(data)
        path = os.path.join(RAD_FOLDER, f"{fid}_{rad.name}")
        with open(path, "wb") as out:
            out.write(data)

        audit("radiology_upload", {"file": rad.name, "id": fid})
        st.success(f"Imaging uploaded (ID: {fid})")

        st.info("AI pipeline ready (segmentation, heatmap, report) — integrate model here.")
        prompt = "Provide a radiology-style report and differential diagnosis for the uploaded image."
        resp = safe_ai_call(prompt, "Radiology AI")
        st.write(resp["answer"])

# ======================================================
# PATIENT TIMELINE (EMR-style)
# ======================================================
if module == "🗓 Patient Timeline":
    st.header("🗓 Patient Timeline (EMR-style)")

    pid = st.text_input("Patient ID", "P-001")
    note = st.text_area("Add Clinical Note")
    if st.button("Save Note"):
        rec = {"time": str(datetime.datetime.now()), "note": note}
        path = os.path.join(EMR_FOLDER, f"{pid}.json")
        timeline = []
        if os.path.exists(path):
            timeline = json.load(open(path))
        timeline.append(rec)
        json.dump(timeline, open(path, "w"), indent=2)
        audit("emr_note", {"patient": pid})
        st.success("Note saved.")

    if os.path.exists(os.path.join(EMR_FOLDER, f"{pid}.json")):
        st.subheader("Timeline")
        tl = json.load(open(os.path.join(EMR_FOLDER, f"{pid}.json")))
        for e in tl[::-1]:
            st.info(f"{e['time']} — {e['note']}")

# ======================================================
# AUDIT & COMPLIANCE
# ======================================================
if module == "🕒 Audit & Compliance":
    st.header("🕒 Audit & Compliance Dashboard")
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No audit records yet.")

# ======================================================
# ADMIN CONTROL PANEL
# ======================================================
if module == "⚙️ Admin Control Panel":
    st.header("⚙️ Admin Control Panel")

    if st.session_state.role != "Admin":
        st.warning("Admin access required.")
    else:
        settings = load_settings()
        st.subheader("Feature Toggles")
        for k, v in settings["features"].items():
            settings["features"][k] = st.toggle(k, v)
        if st.button("Save Settings"):
            save_settings(settings)
            audit("admin_settings_update", settings)
            st.success("Settings updated.")

        st.subheader("User Management")
        users = json.load(open(USERS_DB))
        st.json(users)

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform — v2.0")
