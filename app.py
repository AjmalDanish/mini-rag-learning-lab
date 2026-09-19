"""
Mini RAG Learning Lab — Premium Interactive Dashboard
=====================================================
A Streamlit app that visualises the entire RAG pipeline with a premium UI.

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import os

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini RAG Learning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "A learning-first RAG pipeline — from PDF to vector search to answer."
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a2e;
        --bg-card-hover: #222240;
        --accent-blue: #4f8cff;
        --accent-purple: #a855f7;
        --accent-cyan: #22d3ee;
        --accent-green: #34d399;
        --accent-orange: #fb923c;
        --accent-pink: #f472b6;
        --text-primary: #f0f0f5;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --border: #2a2a3e;
        --gradient-1: linear-gradient(135deg, #4f8cff 0%, #a855f7 100%);
        --gradient-2: linear-gradient(135deg, #22d3ee 0%, #34d399 100%);
        --gradient-3: linear-gradient(135deg, #f472b6 0%, #fb923c 100%);
        --shadow-glow: 0 0 40px rgba(79, 140, 255, 0.15);
    }

    /* Global */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stApp > header { background: transparent !important; }

    /* Main container */
    .main .block-container {
        padding-top: 2rem !important;
        max-width: 1200px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }
    h1 { font-size: 2.5rem !important; letter-spacing: -0.03em !important; }
    h2 { font-size: 1.8rem !important; letter-spacing: -0.02em !important; }
    h3 { font-size: 1.3rem !important; }

    /* Text */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: var(--text-secondary) !important;
        line-height: 1.7 !important;
    }

    /* Hero section */
    .hero-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        background: rgba(79, 140, 255, 0.12);
        border: 1px solid rgba(79, 140, 255, 0.25);
        color: var(--accent-blue);
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em !important;
        line-height: 1.15 !important;
        background: linear-gradient(135deg, #f0f0f5 0%, #9ca3af 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    .hero-subtitle {
        font-size: 1.15rem !important;
        color: var(--text-muted) !important;
        font-weight: 400 !important;
        max-width: 640px !important;
        line-height: 1.6 !important;
    }

    /* Cards */
    .pipeline-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .pipeline-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--gradient-1);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .pipeline-card:hover {
        border-color: rgba(79, 140, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: var(--shadow-glow);
    }
    .pipeline-card:hover::before { opacity: 1; }

    .card-icon {
        width: 48px; height: 48px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        margin-bottom: 0.4rem;
    }
    .card-desc {
        font-size: 0.85rem;
        color: var(--text-muted) !important;
        line-height: 1.5;
    }

    /* Pipeline flow */
    .flow-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem 0;
        overflow-x: auto;
        flex-wrap: wrap;
        justify-content: center;
    }
    .flow-step {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.75rem 1.2rem;
        text-align: center;
        min-width: 120px;
        transition: all 0.3s;
        cursor: default;
    }
    .flow-step:hover {
        border-color: var(--accent-blue);
        background: var(--bg-card-hover);
        transform: scale(1.03);
    }
    .flow-step .step-icon { font-size: 1.4rem; margin-bottom: 0.3rem; }
    .flow-step .step-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-primary) !important;
    }
    .flow-step .step-detail {
        font-size: 0.65rem;
        color: var(--text-muted) !important;
        margin-top: 0.2rem;
    }
    .flow-arrow {
        color: var(--text-muted);
        font-size: 1.2rem;
        flex-shrink: 0;
    }

    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.78rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* Code blocks */
    .stCodeBlock {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }
    .stCodeBlock code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: var(--bg-secondary) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    .stButton > button:hover {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 20px rgba(79, 140, 255, 0.15) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--gradient-1) !important;
        border: none !important;
        color: white !important;
    }

    /* Input */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(79, 140, 255, 0.15) !important;
    }

    /* Dividers */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 2rem 0 !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent-blue) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.6s ease-out forwards; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .pulse { animation: pulse 2s ease-in-out infinite; }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2.5rem 0 1.2rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    .section-header .section-icon {
        width: 36px; height: 36px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
    }
    .section-header .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: var(--gradient-1) !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.8rem;
        padding: 2rem 0;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load the embedding model, cross-encoder, tokenizer, and build the vector DB."""
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from transformers import AutoTokenizer
    import chromadb
    import pymupdf

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model = SentenceTransformer(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cross = CrossEncoder(CROSS_MODEL)

    # Create PDF
    POLICY_TEXT = """COMPANY LEAVE POLICY

Annual Leave:
Employees receive 18 days of annual leave per year.

Sick Leave:
Employees receive 10 days of paid sick leave per year.

Maternity Leave:
Female employees receive 26 weeks of paid maternity leave.

Paternity Leave:
Employees receive 7 days of paid paternity leave.

Work From Home:
Employees may work remotely up to 2 days per week with manager approval.
"""
    pdf_path = "leave_policy.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
    page.insert_textbox(rect, POLICY_TEXT, fontsize=13, fontname="helv")
    doc.save(pdf_path)
    doc.close()

    # Extract text
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_no, page in enumerate(doc, start=1):
        pages.append({
            "text": page.get_text().strip(),
            "metadata": {"source": os.path.basename(pdf_path), "page": page_no},
        })
    doc.close()
    full_text = " ".join(p["text"] for p in pages)

    # Chunk
    def chunk_text(text, chunk_size_words=20, overlap_words=4):
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size_words]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            if i + chunk_size_words >= len(words):
                break
            i += chunk_size_words - overlap_words
        return chunks

    chunks = chunk_text(full_text)

    # Embed
    chunk_embs = model.encode(chunks, normalize_embeddings=True)

    # Vector DB
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="leave_policy", metadata={"hnsw:space": "cosine"}
    )
    ids = [f"chunk_{i:02d}" for i in range(len(chunks))]
    metadatas = [{"source": "leave_policy.pdf", "page": 1, "chunk_index": i} for i in range(len(chunks))]
    collection.add(
        ids=ids, embeddings=chunk_embs.tolist(),
        documents=chunks, metadatas=metadatas,
    )

    return {
        "model": model, "tokenizer": tokenizer, "cross": cross,
        "chunks": chunks, "chunk_embs": chunk_embs,
        "collection": collection, "full_text": full_text,
        "policy_text": POLICY_TEXT,
    }


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def make_flow_step(icon, label, detail=""):
    return f"""
    <div class="flow-step">
        <div class="step-icon">{icon}</div>
        <div class="step-label">{label}</div>
        {f'<div class="step-detail">{detail}</div>' if detail else ''}
    </div>
    """


def make_card(icon, icon_bg, title, desc):
    return f"""
    <div class="pipeline-card">
        <div class="card-icon" style="background: {icon_bg};">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-desc">{desc}</div>
    </div>
    """


def make_metric(value, label):
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Mini RAG Lab")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔬 Pipeline Explorer", "🎯 Live Demo",
         "📊 Analytics", "🎓 Interview Prep"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="padding: 1rem; background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border);">
        <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;">Stack</div>
        <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.8;">
            MiniLM-L6-v2 • ChromaDB<br>
            Cross-Encoder • PyMuPDF<br>
            Sentence Transformers
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (with spinner)
# ──────────────────────────────────────────────────────────────────────────────
if "data" not in st.session_state:
    with st.spinner("Loading models and building vector database..."):
        st.session_state.data = load_models()

data = st.session_state.data


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # Hero
    st.markdown("""
    <div class="fade-in">
        <div class="hero-badge">Open Source • Learning First • No Black Boxes</div>
        <h1 class="hero-title">Mini RAG Learning Lab</h1>
        <p class="hero-subtitle">
            From PDF to vector search to grounded answer — every intermediate value visible.
            Built for deep understanding, not production.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics row
    cols = st.columns(5)
    metrics = [
        ("25", "Sections"),
        ("10", "Embedding Lessons"),
        ("384-d", "Vector Space"),
        ("30,522", "Vocabulary"),
        ("6", "Failure Experiments"),
    ]
    for col, (val, label) in zip(cols, metrics):
        with col:
            st.markdown(make_metric(val, label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline flow
    st.markdown("""
    <div class="section-header">
        <div class="section-icon" style="background: rgba(79, 140, 255, 0.12);">⚡</div>
        <div class="section-title">The Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    flow_html = '<div class="flow-container">'
    steps = [
        ("📄", "PDF", "leave_policy.pdf"),
        ("📖", "Extract", "PyMuPDF"),
        ("✂️", "Chunk", "20 words, 4 overlap"),
        ("🔤", "Tokenize", "WordPiece, 30K vocab"),
        ("📐", "Embed", "MiniLM, 384-d"),
        ("🗄️", "Vector DB", "ChromaDB"),
        ("🔍", "Search", "Cosine similarity"),
        ("🏆", "Re-rank", "Cross-encoder"),
        ("📝", "Prompt", "System + context"),
        ("🤖", "LLM", "Grounded answer"),
    ]
    for i, (icon, label, detail) in enumerate(steps):
        flow_html += make_flow_step(icon, label, detail)
        if i < len(steps) - 1:
            flow_html += '<div class="flow-arrow">→</div>'
    flow_html += '</div>'
    st.markdown(flow_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="section-header">
        <div class="section-icon" style="background: rgba(168, 85, 247, 0.12);">✨</div>
        <div class="section-title">What Makes This Different</div>
    </div>
    """, unsafe_allow_html=True)

    card_cols = st.columns(3)
    features = [
        ("🔬", "rgba(79, 140, 255, 0.12)", "Microscope View",
         "Follow one token through the real MiniLM model — from raw text to the 384 numbers stored in ChromaDB."),
        ("🧪", "rgba(168, 85, 247, 0.12)", "Real Autograd",
         "Watch embedding values update via gradient descent with actual PyTorch autograd — not toy math."),
        ("📊", "rgba(34, 211, 238, 0.12)", "6 Failure Modes",
         "Bad chunks, no overlap, wrong K, noise — each demonstrated with real numbers."),
        ("🎯", "rgba(52, 211, 153, 0.12)", "Cross-Encoder",
         "Bi-encoder retrieval + cross-encoder re-ranking. See before/after scores."),
        ("💬", "rgba(251, 146, 60, 0.12)", "Interview Ready",
         "25 sections with Q&A checkpoints, cheat sheet, and glossary for ML interviews."),
        ("🛠️", "rgba(244, 114, 182, 0.12)", "No LangChain",
         "Every operation hand-built. Then a mapping table shows what LangChain abstracts away."),
    ]
    for col, (icon, bg, title, desc) in zip(card_cols, features):
        with col:
            st.markdown(make_card(icon, bg, title, desc), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Pipeline Explorer":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon" style="background: rgba(79, 140, 255, 0.12);">🔬</div>
        <div class="section-title">Pipeline Explorer</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Text Extraction", "✂️ Chunking", "🔤 Tokenization",
        "📐 Embeddings", "🗄️ Vector DB"
    ])

    with tab1:
        st.markdown("### PDF Text Extraction")
        st.markdown("The raw text extracted from `leave_policy.pdf` using PyMuPDF:")
        st.code(data["policy_text"], language=None)
        st.markdown(f"**Extracted:** {len(data['full_text'])} characters, 1 page")

    with tab2:
        st.markdown("### Chunking Strategy")
        st.markdown(f"**Method:** Sliding window — 20 words, 4-word overlap")
        st.markdown(f"**Chunks produced:** {len(data['chunks'])}")

        for i, c in enumerate(data["chunks"]):
            with st.expander(f"Chunk {i} ({len(c.split())} words)"):
                st.markdown(f"```\n{c}\n```")

        # Chunk overlap visualization
        words = data["full_text"].split()
        chunk_data = []
        for ci, c in enumerate(data["chunks"]):
            cwords = c.split()
            for j in range(len(words) - len(cwords) + 1):
                if words[j:j+3] == cwords[:3]:
                    chunk_data.append({"chunk": ci, "start": j, "length": len(cwords)})
                    break

        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for cd in chunk_data:
            fig.add_trace(go.Bar(
                x=[cd["length"]], y=[f"Chunk {cd['chunk']}"],
                orientation='h', base=cd["start"],
                name=f"Chunk {cd['chunk']}",
                marker_color=colors[cd["chunk"] % len(colors)],
                opacity=0.7,
            ))
        fig.update_layout(
            title="Chunk Coverage of the Document",
            xaxis_title="Word Position",
            yaxis_title="",
            height=250,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width="stretch")

    with tab3:
        st.markdown("### Tokenization")
        st.markdown("The real WordPiece tokenizer of `all-MiniLM-L6-v2`:")

        sample_text = "Female employees receive 26 weeks of paid maternity leave."
        tokens = data["tokenizer"].tokenize(sample_text)
        ids = data["tokenizer"].convert_tokens_to_ids(tokens)

        token_df = pd.DataFrame({"Token": tokens, "Token ID": ids})
        st.dataframe(token_df, width="stretch", hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Vocabulary size:** {len(data['tokenizer'])} tokens")
            st.markdown(f"**Tokens produced:** {len(tokens)}")
        with col2:
            st.markdown("**Subword examples:**")
            for w in ["maternity", "unbelievably", "remotely"]:
                toks = data["tokenizer"].tokenize(w)
                st.markdown(f"  `{w}` → `{toks}`")

    with tab4:
        st.markdown("### Embedding Vectors")
        st.markdown("The real 384-dimensional embedding of the maternity chunk:")

        emb = data["model"].encode("Female employees receive 26 weeks of paid maternity leave.",
                                   normalize_embeddings=True)
        st.markdown(f"**Shape:** {emb.shape}")
        st.markdown(f"**L2 norm:** {np.linalg.norm(emb):.4f} (normalized → direction only)")

        fig = go.Figure(go.Bar(
            x=list(range(48)), y=emb[:48],
            marker_color=["#4f8cff" if v >= 0 else "#f472b6" for v in emb[:48]],
        ))
        fig.update_layout(
            title="First 48 Dimensions of the Chunk Embedding",
            xaxis_title="Dimension",
            yaxis_title="Value",
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width="stretch")

    with tab5:
        st.markdown("### Vector Database (ChromaDB)")
        count = data["collection"].count()
        st.markdown(f"**Collection:** `leave_policy` — **{count} vectors** stored")

        peek = data["collection"].peek(limit=count)
        db_df = pd.DataFrame({
            "ID": peek["ids"],
            "Text": [d[:50] + "..." for d in peek["documents"]],
            "Source": [m["source"] for m in peek["metadatas"]],
        })
        st.dataframe(db_df, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Live Demo":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon" style="background: rgba(52, 211, 153, 0.12);">🎯</div>
        <div class="section-title">Live RAG Demo</div>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Ask a question about the leave policy",
        value="How many weeks of maternity leave are available for a female worker?",
        placeholder="Type your question...",
    )

    k = st.slider("Top-K retrieval", 1, 5, 3)

    if st.button("🔍 Search", type="primary", width="stretch"):
        import torch
        from sklearn.metrics.pairwise import cosine_similarity

        # Step 1: Query embedding
        with st.spinner("Embedding query..."):
            q_emb = data["model"].encode(query, normalize_embeddings=True)

        # Step 2: Vector search
        with st.spinner("Searching vector database..."):
            results = data["collection"].query(
                query_embeddings=[q_emb.tolist()],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        sims = [1.0 - d for d in dists]

        # Step 3: Cross-encoder re-ranking
        with st.spinner("Re-ranking with cross-encoder..."):
            ce_scores = data["cross"].predict([[query, d] for d in docs])
            ce_order = np.argsort(-ce_scores)

        # Display results
        st.markdown("<br>", unsafe_allow_html=True)

        # Query card
        st.markdown(f"""
        <div class="pipeline-card" style="border-left: 3px solid var(--accent-blue);">
            <div class="card-title">📝 Your Query</div>
            <div style="font-size: 1.1rem; color: var(--text-primary); margin-top: 0.5rem;">{query}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Results
        st.markdown("### 🏆 Retrieved & Re-ranked Results")

        for rank, idx in enumerate(ce_order, 1):
            doc = docs[idx]
            meta = metas[idx]
            sim = sims[idx]
            ce = ce_scores[idx]

            # Color based on relevance
            if rank == 1:
                border_color = "var(--accent-green)"
                badge = "🥇 Best Match"
            elif rank == 2:
                border_color = "var(--accent-blue)"
                badge = "🥈 Runner Up"
            else:
                border_color = "var(--border)"
                badge = f"#{rank}"

            st.markdown(f"""
            <div class="pipeline-card" style="border-left: 3px solid {border_color}; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div class="card-title">{badge}</div>
                    <div style="display: flex; gap: 1rem;">
                        <span style="font-size: 0.8rem; color: var(--accent-cyan);">cosine: {sim:.4f}</span>
                        <span style="font-size: 0.8rem; color: var(--accent-purple);">cross-enc: {ce:.2f}</span>
                    </div>
                </div>
                <div style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6;">{doc}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
                    📄 {meta['source']} — page {meta['page']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Comparison chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Scoring Comparison")

        labels = [f"Chunk {i}" for i in range(len(docs))]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Cosine Similarity", "Cross-Encoder Score"))
        fig.add_trace(go.Bar(
            x=labels, y=sims,
            marker_color=["#34d399" if i == ce_order[0] else "#4f8cff" for i in range(len(docs))],
            name="Cosine", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=labels, y=ce_scores,
            marker_color=["#a855f7" if i == ce_order[0] else "#4f8cff" for i in range(len(docs))],
            name="Cross-Encoder", showlegend=False,
        ), row=1, col=2)
        fig.update_layout(
            height=350, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_xaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon" style="background: rgba(251, 146, 60, 0.12);">📊</div>
        <div class="section-title">Embedding Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    # Topic chunks for analysis
    topic_chunks = [
        "Employees receive 18 days of annual leave per year.",
        "Employees receive 10 days of paid sick leave per year.",
        "Female employees receive 26 weeks of paid maternity leave.",
        "Employees receive 7 days of paid paternity leave.",
        "Employees may work remotely up to 2 days per week with manager approval.",
    ]
    topic_labels = ["Annual", "Sick", "Maternity", "Paternity", "WFH"]
    topic_embs = data["model"].encode(topic_chunks, normalize_embeddings=True)

    tab1, tab2, tab3 = st.tabs(["Cosine Matrix", "PCA Projection", "Chunk Similarity"])

    from sklearn.metrics.pairwise import cosine_similarity

    with tab1:
        st.markdown("### Pairwise Cosine Similarity Matrix")
        sim_matrix = cosine_similarity(topic_embs)
        fig = px.imshow(
            sim_matrix, x=topic_labels, y=topic_labels,
            color_continuous_scale="Viridis", zmin=0, zmax=1,
            text_auto=".3f",
        )
        fig.update_layout(
            height=450, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width="stretch")

    with tab2:
        st.markdown("### PCA Projection (384-d → 2-d)")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=0)
        xy = pca.fit_transform(topic_embs)

        fig = px.scatter(
            x=xy[:, 0], y=xy[:, 1], text=topic_labels,
            color=topic_labels,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="top center", marker=dict(size=14))
        fig.update_layout(
            height=450, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"*2D projection preserves {pca.explained_variance_ratio_.sum():.1%} of the original variance.*")

    with tab3:
        st.markdown("### Chunk vs Query Similarity")
        test_query = "How many weeks of maternity leave are available?"
        q_emb = data["model"].encode(test_query, normalize_embeddings=True)
        scores = cosine_similarity(q_emb.reshape(1, -1), topic_embs)[0]

        fig = go.Figure(go.Bar(
            x=topic_labels, y=scores,
            marker_color=["#34d399" if s == scores.max() else "#4f8cff" for s in scores],
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
        ))
        fig.update_layout(
            title=f'Query: "{test_query}"',
            yaxis_title="Cosine Similarity",
            height=350, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 1]),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INTERVIEW PREP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎓 Interview Prep":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon" style="background: rgba(168, 85, 247, 0.12);">🎓</div>
        <div class="section-title">Interview Preparation</div>
    </div>
    """, unsafe_allow_html=True)

    qa_pairs = [
        ("What is RAG and why do we need it?",
         "RAG retrieves relevant external context and injects it into the LLM prompt, so the answer is grounded in up-to-date, domain-specific information. It reduces hallucination and avoids expensive fine-tuning when knowledge changes frequently."),
        ("What is a token ID?",
         "An integer index that identifies a token inside the tokenizer's fixed vocabulary (here 30,522 tokens). It is a lookup position, not a semantic value."),
        ("Where do embedding values come from?",
         "They are learned model parameters, optimized by gradient descent during training: E_new = E_old − lr·dL/dE. No human labels a dimension."),
        ("What is cosine similarity?",
         "(q·c)/(||q||·||c||) — directional similarity between two vectors, in [-1, 1]. It ignores magnitude, which is why embeddings are normalized first."),
        ("Why re-rank with a cross-encoder?",
         "Bi-encoder retrieval is fast but coarse (two independent vectors compared). The cross-encoder reads query and chunk together and re-scores only the top candidates for precision."),
        ("What is self-attention?",
         "Each token gathers information from all tokens, weighted by compatibility: softmax(QK^T/sqrt(d_k))·V. This is what makes representations contextual."),
        ("How does RAG reduce hallucination?",
         "Grounding: the model answers from supplied context under a 'say you don't know' instruction, and the answer carries a source citation."),
        ("RAG vs fine-tuning?",
         "RAG injects editable, citable knowledge at query time; fine-tuning bakes knowledge into weights. RAG suits changing facts, fine-tuning suits behaviour."),
    ]

    for q, a in qa_pairs:
        with st.expander(f"**Q: {q}**"):
            st.markdown(f"**A:** {a}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Cheat sheet
    st.markdown("""
    <div class="section-header">
        <div class="section-icon" style="background: rgba(34, 211, 238, 0.12);">📋</div>
        <div class="section-title">One-Page Cheat Sheet</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ```
    INGESTION                              QUERY
    PDF                                    question
     ↓ PyMuPDF extraction                  ↓ embed with the SAME model
    text + metadata                        query vector
     ↓ chunking (20 words, 4 overlap)      ↓ cosine similarity vs stored vectors
    chunks                                 ↓ top-K retrieval
     ↓ tokenizer (vocab 30,522)            ↓ cross-encoder re-ranking
    tokens → token IDs                     ↓ best chunks
     ↓ embedding-matrix row lookup         ↓ prompt (system + context + question)
    token vectors                          ↓ LLM
     ↓ 12 Transformer layers               ↓ answer + source citation
    contextual token vectors
     ↓ mean pooling + normalize
    384-d chunk embedding
     ↓ store in ChromaDB
    ```
    """)
