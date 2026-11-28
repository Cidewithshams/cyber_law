import os
import re
import requests
import streamlit as st
import gdown
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

st.set_page_config(page_title="CYBER LAW | AI Legal Assistant", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
.stApp { background: #0b1120; color: #f1f5f9; }
.main-title { font-size: 46px; font-weight: 800; text-align: center; color: #38bdf8; }
.subtitle { text-align: center; color: #94a3b8; font-size: 17px; }
div.stButton > button { background: #0284c7; color: white; border-radius: 10px; border: none; width: 100%; height: 48px; font-weight: bold; }
.stTextArea textarea{background:#fff!important;color:#111827!important;}
.stTextArea label{color:#f1f5f9!important}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">⚖️ CYBER LAW</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Pakistan Cyber Law Information Assistant</div>', unsafe_allow_html=True)
st.divider()

api_key = st.secrets.get("GROQ_API_KEY", "") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY", "")
if not api_key:
    st.error("Groq API key missing.")
    st.stop()
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
MODEL = "openai/gpt-oss-20b"

# ---------------- BUILTIN LAWS - NO PDF NEEDED ----------------
BUILTIN_PECA_LAWS = """
TITLE: Prevention of Electronic Crimes Act 2016 Pakistan & Amendment 2025
PECA 2016 - Full Pakistan Cyber Law:
Section 1: Short title extends to whole Pakistan, applies to all citizens.
Section 3: Unauthorized access to information system - punishment 3 months or fine Rs 50,000.
Section 4: Unauthorized copying or transmission of critical data - 6 months or Rs 100,000.
Section 5: Interference with information system - 2 years or Rs 500,000.
Section 6: Unauthorized access to critical infrastructure - 3 years or Rs 1 million.
Section 10: Cyber Terrorism - 14 years imprisonment or Rs 50 million fine.
Section 11: Hate Speech - 7 years or Rs 10 million.
Section 13: Electronic Forgery - 3 years or Rs 250,000.
Section 14: Electronic Fraud - 2 years or Rs 10 million.
Section 15: Making, obtaining or supplying device for offence - 6 months or Rs 50,000.
Section 16: Unauthorized use of identity - 3 years or Rs 5 million.
Section 20: Offence against dignity of natural person - false info harming reputation - 3 years or Rs 1 million. For minors, 5 years.
Section 21: Offence against modesty - intimate images without consent - 5 years or Rs 5 million, for minor 7 years.
Section 22: Child Pornography - 7 years or Rs 5 million, second offence 10 years.
Section 22A: Online grooming and sexual enticement of child - 5 years or Rs 500,000.
Section 24: Cyberstalking - use of email/website to harass - 3 years or Rs 1 million, for minor 5 years.
Section 26A (PECA Amendment 2025 NEW): Intentionally spreading false or fake information likely to cause fear, panic, disorder in public - 3 years imprisonment or Rs 2 million fine or both. 
Section 26B: Authority may block unlawful content within 24 hours.
Authorities: Social Media Protection and Regulatory Authority (SMPRA) established 2025, National Cyber Crime Investigation Agency (NCCIA) replaces FIA Cyber Wing, Social Media Protection Tribunal.
NCCIA Complaint: https://complaint.nccia.gov.pk or helpline 1799 or FIA helpline 9911.
Jurisdiction: NCCIA Courts.
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_from_drive():
    docs, srcs = [], []
    drive_config = st.secrets.get("drive_files", {}) if "drive_files" in st.secrets else {}
    if not drive_config: return [], []
    os.makedirs("data", exist_ok=True)
    for name, file_id in drive_config.items():
        try:
            filepath = f"data/{name}"
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                gdown.download(f"https://drive.google.com/uc?id={file_id}", filepath, quiet=True)
            reader = PdfReader(filepath)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                docs.append(f"TITLE: {name}\n{text}")
                srcs.append(f"https://drive.google.com/file/d/{file_id}/view")
        except: continue
    return docs, srcs

@st.cache_data(ttl=3600, show_spinner=False)
def load_urls():
    path = Path(__file__).parent / "cyber_law_links.txt"
    if not path.exists(): return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]

def extract_content(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=20, verify=False)
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "").lower()
        is_pdf = "application/pdf" in ctype or urlparse(response.url).path.lower().endswith(".pdf")
        if is_pdf:
            reader = PdfReader(BytesIO(response.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text, [], ""
        soup = BeautifulSoup(response.text, "html.parser")
        for el in soup(["script","style","nav","footer","header","noscript"]): el.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        text = soup.get_text(" ", strip=True)
        pdfs = []
        base = urlparse(url).netloc
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            if urlparse(link).path.lower().endswith(".pdf") and urlparse(link).netloc == base:
                pdfs.append(link)
        return f"TITLE: {title}\n{text}", pdfs, ""
    except Exception as e:
        return "", [], str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def build_knowledge_base():
    documents = [BUILTIN_PECA_LAWS]
    sources = ["Built-in PECA 2016 & 2025 Amendment Knowledge Base"]
    errors = []
    try:
        d_docs, d_srcs = load_from_drive()
        documents.extend(d_docs)
        sources.extend(d_srcs)
    except: pass
    try:
        for url in load_urls():
            text, linked, err = extract_content(url)
            if text.strip():
                documents.append(text)
                sources.append(url)
    except: pass
    return documents, sources, errors

def split_text(text, chunk_size=1000, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start+chunk_size].strip()
        if len(chunk) > 80: chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

@st.cache_resource
def create_index(chunks):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=50000)
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix

def retrieve_information(question, chunks, vectorizer, matrix):
    qv = vectorizer.transform([question])
    scores = cosine_similarity(qv, matrix).flatten()
    top = scores.argsort()[-4:][::-1]
    results = []
    for i in top:
        if scores[i] > 0.02:
            results.append({"text": chunks[i], "index": int(i), "score": float(scores[i])})
    return results

def generate_answer(question, context):
    prompt = f"You are CYBER LAW assistant for Pakistan cyber laws. Use ONLY context.\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\nAnswer in simple English with relevant section."
    resp = client.chat.completions.create(model=MODEL, messages=[{"role":"system","content":"You are a careful legal information assistant."},{"role":"user","content":prompt}])
    return resp.choices[0].message.content

with st.sidebar:
    st.title("⚖️ CYBER LAW")
    st.write("No PDF needed - Builtin Law")
    st.divider()
    st.caption("Educational info only.")

st.subheader("Ask a Cyber Law Question")
question = st.text_area("Your Question", placeholder="What does PECA say about online harassment?", height=120)
if st.button("🔍 Generate Legal Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Loading legal sources..."):
            documents, sources, _ = build_knowledge_base()
        chunks = []
        chunk_sources = []
        for doc, src in zip(documents, sources):
            for ch in split_text(doc):
                chunks.append(ch)
                chunk_sources.append(src)
        with st.spinner("Searching..."):
            vectorizer, matrix = create_index(tuple(chunks))
            results = retrieve_information(question, chunks, vectorizer, matrix)
        if not results:
            st.warning("Could not find info.")
        else:
            context = "\n\n".join(r["text"] for r in results)
            used = list(dict.fromkeys(chunk_sources[r["index"]] for r in results))
            with st.spinner("Generating answer..."):
                try:
                    ans = generate_answer(question, context)
                    st.success("Answer Generated")
                    st.markdown("### ⚖️ Legal Information")
                    st.markdown(ans)
                    st.markdown("### 📚 Sources")
                    for s in used: st.markdown(f"- {s}")
                except Exception as e:
                    st.error(f"AI Error: {e}")
st.divider()
st.caption("Disclaimer: Educational info only, not legal advice.")
