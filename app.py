from dotenv import load_dotenv

load_dotenv()

import os
import uuid
from datetime import datetime

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.embeddings import embedding_model
from src.loader import pdf_loader, text_loader
from src.splitter import create_chunks

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Knowledge Mind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_DIR = "data/documents"
MODELS = {
    "GPT-OSS 20B (fast)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (smarter)": "openai/gpt-oss-120b",
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (fastest)": "llama-3.1-8b-instant",
}

SUGGESTIONS = [
    ("Summarize", "Give me a short summary of the uploaded documents."),
    ("Key points", "List the key points I should remember from these documents."),
    ("Find details", "What are the important dates, numbers, or names mentioned?"),
    ("Explain simply", "Explain the main idea of the document in simple words."),
]

PROMPT = ChatPromptTemplate.from_template(
    """You are KnowledgeMind AI, an intelligent document-based question answering assistant.

Answer the user's question using ONLY the information provided below.

RULES:
1. Use the document excerpts as your only source of information.
2. Do not invent, assume, or hallucinate information.
3. If the answer is not available, say: "I couldn't find this information in the uploaded documents."
4. Give a clear, accurate, easy-to-understand answer. Use short bullet points when listing things.
5. If the question needs several pieces of information, combine the relevant parts.
6. If the excerpts conflict, mention the conflict instead of silently choosing one.
7. Do not mention "retrieved context", "chunks", embeddings, or vector databases unless the user asks.
8. When possible, mention the document name and page number the answer came from.
9. Keep the answer concise but complete.
10. If the question is unrelated to the uploaded documents, politely say you can only answer from them.
11. Use the earlier conversation only to understand follow-up questions (like "explain that more").

Earlier conversation:
{chat_history}

Document excerpts:
--------------------
{context}
--------------------

User question:
{question}

Answer:"""
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #14213d;
    --muted: #6b7280;
    --accent: #0e7c7b;
    --accent-soft: #e3f2f1;
    --surface: #ffffff;
    --page: #f6f8f9;
    --line: #e5e9ec;
}

html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'Manrope', system-ui, sans-serif !important;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.stApp { background: var(--page); color: var(--ink); }

/* Make sure text is always readable, even when Streamlit is in dark theme */
.stApp p, .stApp li, .stApp ol, .stApp ul, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6, .stApp label, .stApp strong, .stApp em,
.stApp td, .stApp th, .stApp code,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {
    color: var(--ink);
}
.stApp textarea, .stApp input { color: var(--ink) !important; }
.stApp textarea::placeholder { color: var(--muted) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink); }
.block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 7rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.brand { display: flex; align-items: center; gap: .6rem; margin-bottom: .25rem; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 11px;
    background: var(--accent); color: #fff;
    display: grid; place-items: center; font-size: 1.2rem;
}
.brand-name { font-size: 1.25rem; font-weight: 800; color: var(--ink); line-height: 1.1; }
.brand-sub { font-size: .78rem; color: var(--muted); margin-bottom: 1rem; }

.doc-item {
    display: flex; align-items: center; gap: .6rem;
    padding: .55rem .7rem; margin-bottom: .4rem;
    border: 1px solid var(--line); border-radius: 10px; background: var(--page);
}
.doc-icon { font-size: 1.1rem; }
.doc-name {
    font-size: .85rem; font-weight: 600; color: var(--ink);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 190px;
}
.doc-meta { font-size: .72rem; color: var(--muted); }

.stat-row { display: flex; gap: .5rem; margin: .5rem 0 1rem; }
.stat {
    flex: 1; padding: .6rem .7rem; border-radius: 10px;
    background: var(--accent-soft); text-align: left;
}
.stat b { display: block; font-size: 1.15rem; color: var(--accent); }
.stat span { font-size: .72rem; color: var(--ink); }

/* Hero */
.hero { text-align: center; padding: 3.5rem 0 1.5rem; }
.hero h1 { font-size: 2.4rem; font-weight: 800; color: var(--ink); margin-bottom: .4rem; letter-spacing: -.02em; }
.hero p { color: var(--muted); font-size: 1.02rem; max-width: 520px; margin: 0 auto; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent; border: none; padding: .6rem 0; gap: .9rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse; text-align: left;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--ink); color: #fff; border-radius: 18px 18px 4px 18px;
    padding: .75rem 1.05rem; max-width: 80%; margin-left: auto;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] * { color: #fff; }
.stButton > button[kind="primary"] p, .stButton > button[kind="primary"] * { color: #fff; }
[data-testid="stChatMessageContent"] p { line-height: 1.65; }

/* Source cards */
.src {
    border: 1px solid var(--line); border-left: 3px solid var(--accent);
    background: var(--surface); border-radius: 8px;
    padding: .6rem .8rem; margin-bottom: .5rem;
}
.src-title { font-size: .82rem; font-weight: 700; color: var(--ink); }
.src-text { font-size: .8rem; color: var(--muted); margin-top: .2rem; line-height: 1.5; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 1px solid var(--line);
    font-weight: 600; background: var(--surface); color: var(--ink);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent); color: var(--accent);
}
.stButton > button[kind="primary"] {
    background: var(--accent); border-color: var(--accent); color: #fff;
}
.stButton > button[kind="primary"]:hover { background: #0b6362; color: #fff; }

/* Suggestion buttons */
div[data-testid="stHorizontalBlock"] .stButton > button {
    height: 100%; min-height: 64px; text-align: left; padding: .8rem 1rem;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 16px; border: 1px solid var(--line); background: var(--surface);
    box-shadow: 0 4px 18px rgba(20, 33, 61, .06);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }

@media (max-width: 640px) {
    .hero h1 { font-size: 1.8rem; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] { max-width: 92%; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
defaults = {
    "messages": [],  # {"role", "content", "sources"}
    "vector_db": None,
    "indexed": {},  # file name -> {"chunks": int, "size_kb": float, "type": str}
    "pending_prompt": None,
    "kb_id": uuid.uuid4().hex[:8],
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return embedding_model()


def save_file(uploaded_file) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def load_file(path: str):
    if path.lower().endswith(".pdf"):
        return pdf_loader(file_path=path)
    return text_loader(file_path=path)


def index_files(files):
    """Load, split, embed and add new files to the vector store."""
    new_files = [f for f in files if f.name not in st.session_state.indexed]
    if not new_files:
        st.sidebar.info("These documents are already indexed.")
        return

    with st.sidebar.status("Processing documents...", expanded=True) as status:
        embeddings = get_embeddings()
        if st.session_state.vector_db is None:
            st.session_state.vector_db = Chroma(
                collection_name=f"kb_{st.session_state.kb_id}",
                embedding_function=embeddings,
            )

        for f in new_files:
            try:
                st.write(f"Reading {f.name}")
                path = save_file(f)
                chunks = create_chunks(load_file(path))
                for chunk in chunks:
                    chunk.metadata["file_name"] = f.name
                st.write(f"Embedding {len(chunks)} sections")
                st.session_state.vector_db.add_documents(chunks)
                st.session_state.indexed[f.name] = {
                    "chunks": len(chunks),
                    "size_kb": round(f.size / 1024, 1),
                    "type": f.name.rsplit(".", 1)[-1].upper(),
                }
            except Exception as e:
                st.error(f"Could not process {f.name}: {e}")
        status.update(label="Documents ready", state="complete", expanded=False)


def reset_knowledge_base():
    st.session_state.vector_db = None
    st.session_state.indexed = {}
    st.session_state.messages = []
    st.session_state.kb_id = uuid.uuid4().hex[:8]


def format_history(messages, limit=6) -> str:
    recent = messages[-limit:]
    if not recent:
        return "(none)"
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent
    )


def build_sources(docs):
    sources, seen = [], set()
    for d in docs:
        name = d.metadata.get("file_name") or os.path.basename(d.metadata.get("source", "document"))
        page = d.metadata.get("page")
        page = page + 1 if isinstance(page, int) else None
        key = (name, page, d.page_content[:60])
        if key in seen:
            continue
        seen.add(key)
        snippet = " ".join(d.page_content.split())
        sources.append(
            {
                "name": name,
                "page": page,
                "snippet": snippet[:260] + ("..." if len(snippet) > 260 else ""),
            }
        )
    return sources


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = f" · page {s['page']}" if s["page"] else ""
            st.markdown(
                f"""<div class="src">
                <div class="src-title">📄 {s['name']}{page}</div>
                <div class="src-text">{s['snippet']}</div></div>""",
                unsafe_allow_html=True,
            )


def chat_as_markdown() -> str:
    lines = [f"# Knowledge Mind chat · {datetime.now():%d %b %Y, %H:%M}\n"]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Knowledge Mind"
        lines.append(f"**{who}:**\n\n{m['content']}\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """<div class="brand"><div class="brand-mark">🧠</div>
        <div class="brand-name">Knowledge Mind</div></div>
        <div class="brand-sub">Ask questions, get answers from your own documents</div>""",
        unsafe_allow_html=True,
    )

    if st.button("＋  New chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("#### Documents")
    uploaded = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Process documents", use_container_width=True, disabled=not uploaded):
        index_files(uploaded)

    if st.session_state.indexed:
        total_chunks = sum(d["chunks"] for d in st.session_state.indexed.values())
        st.markdown(
            f"""<div class="stat-row">
            <div class="stat"><b>{len(st.session_state.indexed)}</b><span>documents</span></div>
            <div class="stat"><b>{total_chunks}</b><span>sections</span></div></div>""",
            unsafe_allow_html=True,
        )
        for name, info in st.session_state.indexed.items():
            icon = "📕" if info["type"] == "PDF" else "📝"
            st.markdown(
                f"""<div class="doc-item"><div class="doc-icon">{icon}</div>
                <div><div class="doc-name">{name}</div>
                <div class="doc-meta">{info['type']} · {info['size_kb']} KB · {info['chunks']} sections</div></div></div>""",
                unsafe_allow_html=True,
            )
        if st.button("Remove all documents", use_container_width=True):
            reset_knowledge_base()
            st.rerun()

    with st.expander("Settings"):
        model_label = st.selectbox("Model", list(MODELS.keys()))
        top_k = st.slider("Sections to search", 1, 8, 3, help="More sections give more context but a slower answer.")
        temperature = st.slider("Creativity", 0.0, 1.0, 0.1, 0.05)
        show_sources = st.toggle("Show sources", value=True)

    if st.session_state.messages:
        st.download_button(
            "Download chat",
            data=chat_as_markdown(),
            file_name="knowledge-mind-chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
ready = st.session_state.vector_db is not None

if not st.session_state.messages:
    st.markdown(
        """<div class="hero"><h1>What would you like to know?</h1>
        <p>Upload your documents in the sidebar, then ask anything. Every answer comes with its sources.</p></div>""",
        unsafe_allow_html=True,
    )
    if ready:
        cols = st.columns(2)
        for i, (title, prompt_text) in enumerate(SUGGESTIONS):
            with cols[i % 2]:
                if st.button(f"**{title}**\n\n{prompt_text}", key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt_text
                    st.rerun()
    else:
        st.info("Start by uploading a PDF or text file in the sidebar, then select **Process documents**.")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources:
            render_sources(msg.get("sources"))

# Input
typed = st.chat_input(
    "Ask anything about your documents..." if ready else "Upload and process a document first",
    disabled=not ready,
)
query = typed or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if query and ready:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🧠"):
        try:
            with st.spinner("Searching your documents..."):
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": top_k})
                docs = retriever.invoke(query)
                context = "\n\n".join(
                    f"[{d.metadata.get('file_name', 'document')}"
                    f"{', page ' + str(d.metadata['page'] + 1) if isinstance(d.metadata.get('page'), int) else ''}]\n"
                    f"{d.page_content}"
                    for d in docs
                )

            llm = ChatGroq(model=MODELS[model_label], temperature=temperature)
            chain = PROMPT | llm
            inputs = {
                "context": context,
                "question": query,
                "chat_history": format_history(st.session_state.messages[:-1]),
            }

            answer = st.write_stream(chunk.content for chunk in chain.stream(inputs))
            sources = build_sources(docs)
            if show_sources:
                render_sources(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.caption("Check that GROQ_API_KEY is set in your .env file and that you have internet access.")