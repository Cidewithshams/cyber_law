
import os
import re
import requests
import streamlit as st

from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CYBER LAW | AI Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}
.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    color: #38bdf8;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
}
div.stButton > button {
    background: #0284c7;
    color: white;
    border-radius: 10px;
    border: none;
    width: 100%;
    height: 48px;
    font-weight: bold;
}
.stTextArea textarea{background:#fff!important;color:#111827!important;-webkit-text-fill-color:#111827!important;caret-color:#111827!important}
.stTextArea textarea::placeholder{color:#6b7280!important;-webkit-text-fill-color:#6b7280!important;opacity:1!important}
.stTextArea label{color:#f1f5f9!important}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">⚖️ CYBER LAW</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Pakistan Cyber Law Information Assistant</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- API CONFIG ----------------

api_key = (
    st.secrets.get("GROQ_API_KEY", "")
    if "GROQ_API_KEY" in st.secrets
    else os.getenv("GROQ_API_KEY", "")
)

if not api_key:
    st.error("Groq API key is missing. Add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-20b"


# ---------------- LOAD URLS ----------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_urls():

    path = Path(__file__).parent / "cyber_law_links.txt"

    if not path.exists():
        return []

    urls = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line and not line.startswith("#"):
            urls.append(line)

    return urls


# ---------------- EXTRACT WEB CONTENT ----------------

def extract_content(url):

    headers = {
        "User-Agent": "Mozilla/5.0 CYBER-LAW-Educational-Research-App"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        is_pdf = (
            "application/pdf" in content_type
            or urlparse(response.url).path.lower().endswith(".pdf")
        )

        if is_pdf:
            reader = PdfReader(BytesIO(response.content))

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

            return text, []

        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup([
            "script", "style", "nav",
            "footer", "header", "noscript"
        ]):
            element.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else url

        text = soup.get_text(" ", strip=True)

        # Find official PDF links on the same domain
        pdf_links = []

        base_domain = urlparse(url).netloc

        for anchor in soup.find_all("a", href=True):

            link = urljoin(url, anchor["href"])

            parsed = urlparse(link)

            if (
                parsed.path.lower().endswith(".pdf")
                and parsed.netloc == base_domain
            ):
                pdf_links.append(link)

        return f"TITLE: {title}\n{text}", pdf_links

    except Exception as error:
        st.warning(f"Could not read source: {url} ({error})")
        return "", []


# ---------------- BUILD KNOWLEDGE BASE ----------------

@st.cache_data(ttl=3600, show_spinner=False)
def build_knowledge_base():

    urls = load_urls()

    documents = []
    sources = []
    pdf_count = 0
    max_pdfs = 10

    for url in urls:

        text, linked_pdfs = extract_content(url)

        if text.strip():
            documents.append(text)
            sources.append(url)

        # Read linked PDFs in memory, never save them to disk
        for pdf_url in linked_pdfs:

            if pdf_count >= max_pdfs:
                break

            if pdf_url in sources:
                continue

            pdf_text, _ = extract_content(pdf_url)

            if pdf_text.strip():
                documents.append(pdf_text)
                sources.append(pdf_url)
                pdf_count += 1

    return documents, sources


# ---------------- TEXT CHUNKING ----------------

def split_text(text, chunk_size=1000, overlap=150):

    text = re.sub(r"\s+", " ", text).strip()

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if len(chunk) > 80:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------------- CREATE SEARCH INDEX ----------------

@st.cache_resource
def create_index(chunks):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000
    )

    matrix = vectorizer.fit_transform(chunks)

    return vectorizer, matrix


# ---------------- RETRIEVAL ----------------

def retrieve_information(question, chunks, vectorizer, matrix):

    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = scores.argsort()[-4:][::-1]

    results = []

    for index in top_indices:

        if scores[index] > 0.02:

            results.append({
                "text": chunks[index],
                "score": float(scores[index])
            })

    return results


# ---------------- AI RESPONSE ----------------

def generate_answer(question, context):

    prompt = f"""
You are CYBER LAW, an educational AI assistant
specialized in Pakistan's cyber laws.

Answer the user's question using ONLY the legal
information supplied in the retrieved context.

Rules:
1. Do not invent legal sections or penalties.
2. Explain legal concepts in simple English.
3. Mention the relevant act or section only when
   supported by the provided context.
4. If the context does not contain the answer,
   clearly say that you could not verify it.
5. Never pretend to be a lawyer.
6. Distinguish general information from legal advice.
7. Include source URLs in your answer when relevant.

RETRIEVED LEGAL CONTEXT:

{context}

USER QUESTION:

{question}

Give a clear and helpful answer.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a careful legal information assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("⚖️ CYBER LAW")

    st.write("AI Legal Information Assistant")

    st.divider()

    st.markdown("### Features")

    st.write("🌐 Official website sources")
    st.write("🔎 RAG-based retrieval")
    st.write("🤖 GPT-OSS AI model")
    st.write("📚 Source-based answers")

    st.divider()

    st.markdown("### Legal Coverage")

    st.write("Pakistan Cyber Laws")
    st.write("PECA 2016")
    st.write("PECA Amendment 2025")

    st.divider()

    st.caption("Educational information only.")


# ---------------- MAIN APPLICATION ----------------

st.subheader("Ask a Cyber Law Question")

st.write(
    "Enter a question about cybercrime, online harassment, "
    "identity theft, digital fraud or Pakistan's cyber laws."
)

question = st.text_area(
    "Your Question",
    placeholder="Example: What does PECA say about online harassment?",
    height=120
)

if st.button("🔍 Generate Legal Answer"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Loading official legal sources..."):

            documents, sources = build_knowledge_base()

        if not documents:

            st.error("Could not retrieve legal information from websites.")

        else:

            chunks = []

            chunk_sources = []

            for document, source in zip(documents, sources):

                for chunk in split_text(document):

                    chunks.append(chunk)
                    chunk_sources.append(source)

            if not chunks:

                st.error("No readable legal text was found.")

            else:

                with st.spinner("Searching legal knowledge base..."):

                    vectorizer, matrix = create_index(tuple(chunks))

                    results = retrieve_information(
                        question,
                        chunks,
                        vectorizer,
                        matrix
                    )

                if not results:

                    st.warning(
                        "I could not find sufficient supporting "
                        "information in the available sources. "
                        "Please check the official legal sources."
                    )

                else:

                    context = "\n\n".join(
                        result["text"]
                        for result in results
                    )

                    used_sources = list(dict.fromkeys(
                        chunk_sources[
                            chunks.index(result["text"])
                        ]
                        for result in results
                    ))

                    with st.spinner("GPT-OSS is preparing your answer..."):

                        try:

                            answer = generate_answer(
                                question,
                                context
                            )

                            st.success("Answer Generated")

                            st.markdown("### ⚖️ Legal Information")

                            st.markdown(answer)

                            st.markdown("### 📚 Retrieved Sources")

                            for source in used_sources:

                                st.markdown(f"- [{source}]({source})")

                        except Exception as error:

                            st.error(f"AI API Error: {error}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "CYBER LAW | AI-Powered Legal Information Assistant"
)

st.caption(
    "Disclaimer: This application provides educational "
    "information, not professional legal advice. "
    "Always verify laws against official Gazette publications."
)