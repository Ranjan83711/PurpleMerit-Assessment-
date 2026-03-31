"""
streamlit_app.py — KUK Course Planning Assistant UI

Run FROM project root: streamlit run app/streamlit_app.py
OR from app/ folder:   streamlit run streamlit_app.py
"""
import os
import sys
import json
import time

# ── Resolve project root regardless of CWD ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)          # ensures all relative paths (data/, vectorstore/) work

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "env", ".env"))

import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="KUK Course Planning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2rem; margin: 0; font-weight: 700; }
    .main-header p  { color: #a0aec0; margin: 0.5rem 0 0; font-size: 0.95rem; }

    /* Chat bubbles */
    .user-bubble {
        background: #1e3a5f;
        border: 1px solid #2d5986;
        border-radius: 12px 12px 2px 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
        max-width: 80%;
        margin-left: auto;
        color: #e8f4fd !important;
    }
    .assistant-bubble {
        background: #1a2332;
        border: 1px solid #2d3748;
        border-radius: 12px 12px 12px 2px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
        max-width: 90%;
        color: #e2e8f0 !important;
    }
    .user-bubble * , .assistant-bubble * { color: inherit !important; }

    /* Section cards */
    .section-card {
        background: #1a2332;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    .section-card h4 {
        color: #90cdf4;
        margin: 0 0 0.5rem;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Decision badge */
    .badge-eligible   { background:#c6f6d5; color:#22543d; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
    .badge-not-eligible { background:#fed7d7; color:#742a2a; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
    .badge-need-info  { background:#fef3c7; color:#744210; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
    .badge-abstain    { background:#e2e8f0; color:#2d3748; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }

    /* Citation tag */
    .citation-tag {
        display: inline-block;
        background: #ebf8ff;
        color: #2b6cb0;
        border: 1px solid #bee3f8;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        margin: 2px;
        font-family: monospace;
    }

    /* Sidebar */
    .sidebar-section {
        background: #f7fafc;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }

    /* Example queries */
    .example-query {
        background: #f0fff4;
        border: 1px solid #9ae6b4;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        cursor: pointer;
    }

    /* Status indicator */
    .status-ok   { color: #38a169; font-weight: 600; }
    .status-warn { color: #d69e2e; font-weight: 600; }
    .status-err  { color: #e53e3e; font-weight: 600; }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "crew" not in st.session_state:
        st.session_state.crew = None
    if "index_built" not in st.session_state:
        st.session_state.index_built = False
    if "student_profile" not in st.session_state:
        st.session_state.student_profile = {
            "completed_courses": [],
            "target_program": "",
            "target_term": "Spring 2026",
            "max_credits": 20,
            "grades": {},
        }


init_session()


# ─────────────────────────────────────────────
# LOADING HELPERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load LLM, embeddings, vector store, and crew. Cached across sessions."""
    from configs.model_config import get_groq_llm, get_embedding_model
    from rag.vector_store import CourseVectorStore
    from rag.retriever import CatalogRetriever
    from crew.crew_setup import CoursePlanningCrew

    llm = get_groq_llm()
    embeddings = get_embedding_model()
    vs = CourseVectorStore(
        persist_directory=os.getenv("CHROMA_DB_PATH", "vectorstore/chroma_db"),
        collection_name=os.getenv("COLLECTION_NAME", "kuk_course_catalog"),
    )
    vs.load(embeddings)
    retriever = CatalogRetriever(vs, k=int(os.getenv("RETRIEVER_K", 6)))
    pdf_path = os.path.join(PROJECT_ROOT, os.getenv("PDF_PATH", "data/raw/kuk_prospectus_2011.pdf"))
    crew = CoursePlanningCrew(llm=llm, retriever=retriever, pdf_path=pdf_path)
    return crew, vs


def build_index_ui():
    """Build the vector index from the PDF (shown in sidebar)."""
    from rag.loader import CatalogLoader
    from rag.cleaner import TextCleaner
    from rag.chunker import CatalogChunker
    from rag.vector_store import CourseVectorStore
    from configs.model_config import get_embedding_model

    pdf_path = os.getenv("PDF_PATH", "data/raw/kuk_prospectus_2011.pdf")
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found at `{pdf_path}`. Please upload it to `data/raw/`.")
        return False

    progress = st.progress(0, text="Loading PDF...")
    try:
        loader = CatalogLoader(pdf_path)
        raw_docs = loader.load()
        progress.progress(25, text=f"Loaded {len(raw_docs)} pages. Cleaning...")

        cleaner = TextCleaner()
        clean_docs = cleaner.clean(raw_docs)
        progress.progress(50, text=f"Cleaned {len(clean_docs)} pages. Chunking...")

        chunker = CatalogChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", 800)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 150)),
        )
        chunks = chunker.chunk(clean_docs)
        progress.progress(75, text=f"Created {len(chunks)} chunks. Building index...")

        embeddings = get_embedding_model()
        vs = CourseVectorStore(
            persist_directory=os.getenv("CHROMA_DB_PATH", "vectorstore/chroma_db"),
            collection_name=os.getenv("COLLECTION_NAME", "kuk_course_catalog"),
        )
        vs.build(chunks, embeddings)
        progress.progress(100, text="✅ Index built!")
        time.sleep(1)
        progress.empty()
        st.session_state.index_built = True
        return True
    except Exception as e:
        progress.empty()
        st.error(f"Index build failed: {e}")
        return False


def check_index_exists() -> bool:
    chroma_path = os.path.join(
        os.getenv("CHROMA_DB_PATH", "vectorstore/chroma_db"), "chroma.sqlite3"
    )
    return os.path.exists(chroma_path)


# ─────────────────────────────────────────────
# RESPONSE RENDERER
# ─────────────────────────────────────────────
def render_response(result: dict):
    """Render a structured crew response in the chat."""
    raw = result.get("raw_output", "")
    sections = result.get("sections", {})
    decision = result.get("decision", "UNKNOWN")

    # Decision badge
    badge_map = {
        "ELIGIBLE": "badge-eligible",
        "NOT ELIGIBLE": "badge-not-eligible",
        "NOT_ELIGIBLE": "badge-not-eligible",
        "NEED MORE INFO": "badge-need-info",
        "NEED_MORE_INFO": "badge-need-info",
        "CANNOT DETERMINE": "badge-abstain",
        "APPROVED": "badge-eligible",
    }
    badge_class = badge_map.get(decision.upper(), "badge-need-info")
    st.markdown(
        f'<span class="{badge_class}">{decision}</span>', unsafe_allow_html=True
    )

    # Answer / Plan
    if sections.get("answer"):
        with st.expander("📋 Answer / Plan", expanded=True):
            st.markdown(sections["answer"])

    # Why
    if sections.get("why"):
        with st.expander("🔍 Reasoning (Prerequisites / Requirements)", expanded=True):
            st.markdown(sections["why"])

    # Citations
    if sections.get("citations"):
        with st.expander("📚 Citations", expanded=True):
            citation_lines = [
                line.strip()
                for line in sections["citations"].split("\n")
                if line.strip() and line.strip() not in ("-", "*")
            ]
            for cite in citation_lines:
                st.markdown(
                    f'<span class="citation-tag">📄 {cite}</span>',
                    unsafe_allow_html=True,
                )

    # Clarifying Questions
    if sections.get("clarifying_questions") and sections["clarifying_questions"].strip() not in ("None", "none", "N/A", ""):
        with st.expander("❓ Clarifying Questions", expanded=True):
            st.info(sections["clarifying_questions"])

    # Assumptions / Not in catalog
    if sections.get("assumptions") and sections["assumptions"].strip() not in ("None", "none", "N/A", ""):
        with st.expander("⚠️ Assumptions / Not in Catalog", expanded=False):
            st.warning(sections["assumptions"])

    # Verification
    if sections.get("verification"):
        with st.expander("✅ Verification Report", expanded=False):
            verif = sections["verification"]
            if "PASS" in verif.upper():
                st.success(verif)
            elif "FAIL" in verif.upper() or "NEEDS_REVISION" in verif.upper():
                st.warning(verif)
            else:
                st.info(verif)

    # Fallback: raw output
    if not any(sections.values()):
        st.markdown(raw)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 KUK Course Planner")
    st.markdown("*Powered by CrewAI + Groq + ChromaDB*")
    st.divider()

    # Index status
    index_ok = check_index_exists()
    if index_ok:
        st.markdown('<span class="status-ok">✅ Index Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">❌ Index Not Built</span>', unsafe_allow_html=True)
        st.caption("Build the index before chatting.")

    if st.button("🔧 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Building index from KUK catalog PDF..."):
            success = build_index_ui()
            if success:
                st.success("Index built! Reload the page to start chatting.")
                st.cache_resource.clear()

    st.divider()

    # Student Profile (pre-fills queries)
    with st.expander("👤 Student Profile (Optional)", expanded=False):
        program = st.text_input(
            "Target Program",
            value=st.session_state.student_profile["target_program"],
            placeholder="e.g., B.Tech Computer Science",
        )
        term = st.selectbox(
            "Target Term",
            ["Spring 2026", "Fall 2026", "Annual 2026", "Spring 2027"],
            index=0,
        )
        max_credits = st.number_input(
            "Max Credits / Term",
            min_value=10, max_value=40, value=20, step=1,
        )
        completed = st.text_area(
            "Completed Courses (one per line)",
            placeholder="Mathematics-I\nPhysics-I\nEngineering Drawing",
            height=100,
        )

        if st.button("Save Profile", use_container_width=True):
            st.session_state.student_profile = {
                "target_program": program,
                "target_term": term,
                "max_credits": max_credits,
                "completed_courses": [
                    c.strip() for c in completed.split("\n") if c.strip()
                ],
            }
            st.success("Profile saved!")

    st.divider()

    # Example queries
    st.markdown("**💡 Example Queries**")
    examples = [
        "Can I take Digital Electronics if I've completed Analog Electronics?",
        "What courses can I take next semester? I've completed 1st year.",
        "What are the prerequisites for Computer Networks?",
        "What is the total credit requirement for B.Tech?",
        "Which professor teaches DSP this semester?",
    ]
    for ex in examples:
        if st.button(ex[:55] + ("..." if len(ex) > 55 else ""), use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["prefill_query"] = ex

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("**Sources:** KUK Prospectus / Academic Catalog")
    st.caption("**Model:** Groq llama-3.3-70b-versatile")
    st.caption("**Vector DB:** ChromaDB (cosine similarity)")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 KUK Course Planning Assistant</h1>
    <p>Prerequisite checks · Course plans · Academic policy — grounded in the official catalog</p>
</div>
""", unsafe_allow_html=True)

# Mode selector
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📋 Plan My Semester", use_container_width=True, type="primary"):
        profile = st.session_state.student_profile
        courses_str = ", ".join(profile["completed_courses"]) if profile["completed_courses"] else "none yet"
        st.session_state["prefill_query"] = (
            f"I am pursuing {profile['target_program'] or 'B.Tech'}. "
            f"I have completed: {courses_str}. "
            f"Please suggest courses for {profile['target_term']} (max {profile['max_credits']} credits)."
        )
with col2:
    if st.button("✅ Check Eligibility", use_container_width=True):
        st.session_state["prefill_query"] = (
            "Can I take [COURSE NAME] if I have completed [LIST YOUR COURSES]?"
        )
with col3:
    if st.button("📖 Lookup Prerequisites", use_container_width=True):
        st.session_state["prefill_query"] = (
            "What are all the prerequisites to take [COURSE NAME]?"
        )

st.divider()

# Chat history display
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑‍🎓 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.container():
            st.markdown('<div class="assistant-bubble">', unsafe_allow_html=True)
            st.markdown("**🎓 Course Planning Assistant**")
            if isinstance(msg["content"], dict):
                render_response(msg["content"])
            else:
                st.markdown(msg["content"])
            st.markdown("</div>", unsafe_allow_html=True)

# Query input
prefill = st.session_state.pop("prefill_query", "")
query = st.chat_input(
    "Ask about prerequisites, plan your semester, or look up any academic policy...",
)
if prefill and not query:
    query = prefill

# Process query
if query:
    # Gate: check index
    if not check_index_exists():
        st.error("⚠️ Please build the index first using the sidebar button.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(
        f'<div class="user-bubble">🧑‍🎓 {query}</div>', unsafe_allow_html=True
    )

    # Load pipeline
    with st.spinner("🤖 Agents working..."):
        try:
            if st.session_state.crew is None:
                crew, _ = load_pipeline()
                st.session_state.crew = crew
            else:
                crew = st.session_state.crew

            # Build enriched query with student profile if available
            profile = st.session_state.student_profile
            enriched_query = query
            if profile["completed_courses"] or profile["target_program"]:
                context_parts = []
                if profile["target_program"]:
                    context_parts.append(f"Program: {profile['target_program']}")
                if profile["completed_courses"]:
                    context_parts.append(
                        f"Completed courses: {', '.join(profile['completed_courses'])}"
                    )
                if profile["target_term"]:
                    context_parts.append(f"Target term: {profile['target_term']}")
                enriched_query = query + "\n\n[Student context: " + " | ".join(context_parts) + "]"

            # Route to appropriate pipeline
            lower = query.lower()
            if any(
                k in lower
                for k in ["plan", "schedule", "semester", "next term", "what courses can i"]
            ):
                result = crew.run_course_plan(enriched_query)
            elif any(
                k in lower
                for k in ["can i take", "eligible", "am i eligible", "qualify", "have i met"]
            ):
                result = crew.run_eligibility_check(enriched_query)
            else:
                result = crew.run_general_query(enriched_query)

            # Display response
            with st.container():
                st.markdown('<div class="assistant-bubble">', unsafe_allow_html=True)
                st.markdown("**🎓 Course Planning Assistant**")
                render_response(result)
                st.markdown("</div>", unsafe_allow_html=True)

            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": result})

        except ValueError as ve:
            err_msg = str(ve)
            if "GROQ_API_KEY" in err_msg:
                st.error(
                    "**API Key Missing** — Add your `GROQ_API_KEY` to `env/.env`.\n\n"
                    "Get a free key at [console.groq.com](https://console.groq.com)"
                )
            else:
                st.error(f"Configuration error: {ve}")
        except FileNotFoundError as fe:
            st.error(
                f"**File not found:** {fe}\n\n"
                "Make sure your KUK catalog PDF is at `data/raw/kuk_prospectus_2011.pdf`"
            )
        except Exception as e:
            st.error(f"**Error:** {str(e)[:300]}")
            st.caption("If this is a rate limit error, please wait a moment and try again.")