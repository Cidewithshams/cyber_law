
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

/* FIX: Make question input text clearly visible */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
    border: 1px solid #94a3b8 !important;
}

.stTextArea textarea::placeholder {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
    opacity: 1 !important;
}

.stTextArea label {
    color: #f1f5f9 !important;
}

.stTextInput label,
.stSelectbox label {
    color: #f1f5f9 !important;
}
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
