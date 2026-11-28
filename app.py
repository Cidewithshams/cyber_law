import os, re, requests
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

st.set_page_config(page_title="CYBER LAW • Pakistan's Legal AI", page_icon="⚖️", layout="wide")

# Hide Streamlit branding + fix empty box
st.markdown("<style>#stDecoration, header, footer, .stDeployButton, [data-testid='stManageAppButton'], div[data-testid='stToolbar']{display:none!important} .block-container{padding-top:1rem!important; max-width: 1100px;} div[data-testid='stVerticalBlock'] > div:empty { display: none !important; }</style>", unsafe_allow_html=True)

# ================= PROFESSIONAL UI CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp {
  background: 
    radial-gradient(1200px 600px at 10% -10%, rgba(56,189,248,0.18), transparent),
    radial-gradient(900px 500px at 100% 0%, rgba(139,92,246,0.18), transparent),
    radial-gradient(800px 600px at 50% 120%, rgba(16,185,129,0.12), transparent),
    #070D1E;
}
.header-wrap { text-align: center; padding: 12px 0 8px 0; }
.logo { font-size: 44px; font-weight: 800; background: linear-gradient(92deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; text-align:center; line-height:1.1 }
.sub { color: #94A3B8; font-size: 12px; letter-spacing: 3px; text-transform: uppercase; margin-top: 6px; font-weight: 600; }
.badge { display: inline-block; margin-top: 10px; background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.2); color: #7DD3FC; padding: 6px 14px; border-radius: 100px; font-size: 11px; }
.q-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.03) 100%);
  backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px; padding: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); margin-top:12px;
}
.q-label { color: #E2E8F0; font-weight: 700; font-size: 13px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; letter-spacing:1px }
.stTextArea textarea {
  background: rgba(7,13,30,0.8)!important; color: #F1F5F9!important;
  border: 1.5px solid rgba(56,189,248,0.25)!important; border-radius: 14px!important;
  font-size: 15px!important; padding: 14px!important; min-height: 110px!important;
}
.stTextArea textarea:focus { border-color: #38bdf8!important; box-shadow: 0 0 0 3px rgba(56,189,248,0.15)!important; }
div.stButton > button {
  background: linear-gradient(100deg, #0EA5E9 0%, #6366F1 100%);
  color: white; border: none; border-radius: 14px; height: 56px; font-weight: 800;
  font-size: 16px; letter-spacing: 0.5px; box-shadow: 0 12px 30px rgba(14,165,233,0.35);
}
div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 18px 40px rgba(14,165,233,0.45); }
.legal-card {
  background: linear-gradient(180deg, rgba(56,189,248,0.12) 0%, rgba(56,189,248,0.03) 100%);
  border: 1px solid rgba(56,189,248,0.2); border-left: 4px solid #38bdf8;
  border-radius: 16px; padding: 20px; margin-top: 18px;
}
.consequence-card {
  background: linear-gradient(180deg, rgba(251,146,60,0.12) 0%, rgba(251,146,60,0.03) 100%);
  border: 1px solid rgba(251,146,60,0.25); border-left: 4px solid #FB923C;
  border-radius: 16px; padding: 20px; margin-top: 14px;
}
.source-card {
  background: linear-gradient(180deg, rgba(16,185,129,0.1) 0%, rgba(16,185,129,0.02) 100%);
  border: 1px solid rgba(16,185,129,0.2); border-left: 4px solid #10B981;
  border-radius: 16px; padding: 16px; margin-top: 14px;
}
.source-pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.25);
  color: #6EE7B7; padding: 7px 14px; border-radius: 100px; font-size: 12px; margin: 4px; text-decoration: none;
}
.card-title { font-weight: 800; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 10px; }
.quick-btns { display:flex; gap:8px; flex-wrap:wrap; margin: 10px 0 6px 0; }
</style>
<div class="header-wrap">
  <div class="logo">⚖️ CYBER LAW</div>
  <div class="sub">Pakistan's Official Cyber Legal Intelligence</div>
  <div class="badge">● LIVE • NCCIA • PECA 2016 & 2025 Amendment • Official Sources</div>
</div>
""", unsafe_allow_html=True)

# ================= BACKEND =================
api_key = st.secrets.get("GROQ_API_KEY","") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY","")
if not api_key:
    st.error("GROQ_API_KEY missing in secrets")
    st.stop()
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
MODEL = "openai/gpt-oss-20b"

BUILTIN = "PECA 2016 & 2025: Sec3 unauthorized access 3mo 50k, Sec4 copy 6mo 100k, Sec5 interference 2y 500k, Sec10 cyber terrorism 14y 50M, Sec16 identity theft 3y 5M, Sec18 mobile IMEI tamper 3y 5Lakh, Sec20 dignity 3y 1M, Sec21 modesty intimate images 5y 5M, Sec22 child porn 7y, Sec24 cyberstalking 3y 1M, Sec26A 2025 fake info panic 3y 2M. NCCIA portal complaint.nccia.gov.pk helpline 1799 replaces FIA Cybercrime."

@st.cache_data(ttl=3600, show_spinner=False)
def load_urls():
    p = Path(__file__).parent / "cyber_law_links.txt"
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")] if p.exists() else []

def extract_content(url):
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=25, verify=False)
        r.raise_for_status()
        if "application/pdf" in r.headers.get("Content-Type","").lower() or url.lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(r.content))
            txt = "\n".join(page.extract_text() or "" for page in reader.pages)
            return txt, []
        soup = BeautifulSoup(r.text, "html.parser")
        for el in soup(["script","style","nav","footer"]): el.decompose()
        return f"TITLE:{soup.title.get_text() if soup.title else url}\n{soup.get_text(' ', strip=True)}", []
    except: return "", []

@st.cache_data(ttl=3600, show_spinner=False)
def build_kb():
    docs=[BUILTIN]; srcs=["Built-in PECA Knowledge"]
    for url in load_urls():
        t,_=extract_content(url)
        if len(t)>200: docs.append(t); srcs.append(url)
    return docs, srcs

def split_text(t, s=1000, o=150):
    t=re.sub(r"\s+"," ",t).strip()
    return [t[i:i+s] for i in range(0,len(t),s-o) if len(t[i:i+s].strip())>80]

@st.cache_resource
def create_index(chunks):
    v=TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=50000)
    m=v.fit_transform(chunks)
    return v,m

def retrieve(q,chunks,v,m):
    qv=v.transform([q])
    scores=cosine_similarity(qv,m).flatten()
    top=scores.argsort()[-5:][::-1]
    return [{"text":chunks[i],"idx":int(i),"score":float(scores[i])} for i in top if scores[i]>0.01]

def gen_ans(q,ctx):
    prompt=f"You are CYBER LAW expert for Pakistan. Context:\n{ctx}\n\nQ:{q}\nAnswer in 2 parts: 1) Legal Info with PECA section, 2) Consequences with years/fine. Simple English, professional."
    r=client.chat.completions.create(model=MODEL, messages=[{"role":"system","content":"You are Pakistan cyber law expert"},{"role":"user","content":prompt}], temperature=0.2)
    return r.choices[0].message.content

# ================= UI =================
# Quick chips
if 'question_prefill' not in st.session_state:
    st.session_state['question_prefill'] = ""

c1,c2,c3 = st.columns(3)
if c1.button("💡 FB Hacked?", use_container_width=True): st.session_state['question_prefill'] = "My Facebook account hacked, which PECA section applies?"
if c2.button("📱 IMEI Tamper?", use_container_width=True): st.session_state['question_prefill'] = "What is punishment for mobile IMEI tampering under Section 18?"
if c3.button("🚨 Fake News Law?", use_container_width=True): st.session_state['question_prefill'] = "What is Section 26A fake news law 2025 punishment?"

st.markdown('<div class="q-card">', unsafe_allow_html=True)
st.markdown('<div class="q-label">💬 ASK YOUR LEGAL QUESTION</div>', unsafe_allow_html=True)
question = st.text_area("", value=st.session_state['question_prefill'], placeholder="e.g. What is punishment for mobile hacking under PECA? • How to file NCCIA complaint? • What is Section 26A fake news law?", height=115, label_visibility="collapsed")
btn = st.button("⚡ GENERATE LEGAL ANSWER", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if btn:
    if not question.strip(): st.warning("Please type a question")
    else:
        with st.spinner("🔍 Searching Official NCCIA & PECA Sources..."):
            docs,srcs=build_kb()
            chunks=[]; csrc=[]
            for d,s in zip(docs,srcs):
                for ch in split_text(d): chunks.append(ch); csrc.append(s)
            v,m=create_index(tuple(chunks))
            res=retrieve(question,chunks,v,m)
        if not res: st.error("Not found in official sources")
        else:
            ctx="\n\n".join(r["text"] for r in res)
            used=list(dict.fromkeys([csrc[r["idx"]] for r in res]))
            with st.spinner("⚖️ Drafting Professional Answer..."):
                ans=gen_ans(question,ctx)
            
            st.markdown(f'<div class="legal-card"><div class="card-title" style="color:#38bdf8">⚖️ LEGAL INFORMATION</div><div style="color:#E2E8F0; line-height:1.7">{ans}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="consequence-card"><div class="card-title" style="color:#FB923C">🚨 CONSEQUENCES & PUNISHMENT</div><div style="color:#FED7AA; font-size:14px">As per PECA official gazette: Check detailed section above. For Fake News Sec 26A: 3 years / Rs 2M. For IMEI tamper Sec 18: 3 years / Rs 5 Lakh. For Child offences Sec 22: up to 7 years.</div></div>', unsafe_allow_html=True)
            pills="".join([f'<a class="source-pill" href="{u}" target="_blank">📄 {urlparse(u).netloc}</a>' if "http" in u else f'<span class="source-pill">📚 {u}</span>' for u in used])
            st.markdown(f'<div class="source-card"><div class="card-title" style="color:#10B981">📚 VERIFIED OFFICIAL SOURCES</div>{pills}<div style="margin-top:10px; color:#6B7280; font-size:11px">Sources: NCCIA Laws, Pakistan Code, National Assembly Gazette</div></div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="margin-top:18px; text-align:center">
            <a href="https://complaint.nccia.gov.pk" target="_blank" style="display:inline-block; background: linear-gradient(90deg,#10b981,#059669); color:white; padding:12px 26px; border-radius:100px; text-decoration:none; font-weight:800; font-size:13px; box-shadow: 0 8px 20px rgba(16,185,129,0.3)">🚨 File Complaint Directly on NCCIA Portal →</a>
            <div style="margin-top:8px; color:#64748b; font-size:11px">Helpline: 1799 | complaint.nccia.gov.pk</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div style="text-align:center; margin-top:32px; color:#475569; font-size:11px">© CYBER LAW • Educational Information Only • Not Legal Advice • Data from NCCIA Official Portal</div>', unsafe_allow_html=True)
