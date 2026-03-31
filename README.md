# 🎓 KUK Course Planning Assistant — Agentic RAG (Assessment 1)

**AI/ML Engineer Intern Assessment — Purple Merit Technologies**  
Built with: **CrewAI · Groq (llama-3.3-70b-versatile) · ChromaDB · Streamlit**

---

## 📐 Architecture Overview

```
Student Query
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                     CrewAI Pipeline                          │
│                                                              │
│  [1] Intake Agent          ← Normalizes student profile     │
│       │                      Asks clarifying questions       │
│       ▼                                                      │
│  [2] Retriever Agent  ←── ChromaDB (cosine similarity)      │
│       │                   KUK Catalog chunks + citations     │
│       ▼                                                      │
│  [3] Rule Extractor Agent  ← Parses prereq chains (AND/OR)  │
│       │                      Grade reqs, co-reqs, exceptions │
│       ▼                                                      │
│  [4] Planner Agent         ← Generates cited course plan    │
│       │                      or eligibility decision         │
│       ▼                                                      │
│  [5] Verifier Agent        ← Citation audit + logic check   │
│                               PASS / NEEDS_REVISION         │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Structured Output:
  Answer/Plan · Why · Citations · Clarifying Qs · Assumptions
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/ai-course-planner-rag.git
cd ai-course-planner-rag
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Edit env/.env
GROQ_API_KEY=your_groq_api_key_here   # Free at console.groq.com
```

### 3. Add KUK Catalog PDF

Place the KUK prospectus PDF at:
```
data/raw/kuk_prospectus_2011.pdf
```

### 4. Build the Index

```bash
python main.py --build-index
```

### 5. Launch the UI

```bash
streamlit run app/streamlit_app.py
```

Or use CLI:
```bash
python main.py --chat
python main.py --query "Can I take Digital Electronics if I've completed Analog Electronics?"
```

---

## 🗂️ Project Structure

```
ai-course-planner-rag/
├── data/
│   ├── raw/                    # KUK catalog PDF (you provide)
│   ├── processed/              # Cleaned text + chunks (auto-generated)
│   └── evaluation/             # 25 test queries + results
│
├── vectorstore/
│   └── chroma_db/              # ChromaDB persistent index (auto-generated)
│
├── rag/                        # RAG Pipeline
│   ├── loader.py               # PDF ingestion (pdfplumber + pypdf fallback)
│   ├── cleaner.py              # Text normalization
│   ├── chunker.py              # RecursiveCharacterTextSplitter
│   ├── embedder.py             # all-MiniLM-L6-v2 embeddings
│   ├── vector_store.py         # ChromaDB build/load/search
│   ├── retriever.py            # Similarity search + citation formatting
│   └── prompt_templates.py     # All agent prompts
│
├── agents/                     # CrewAI Agents
│   ├── intake_agent.py         # Student profile normalization
│   ├── retriever_agent.py      # Catalog search specialist
│   ├── rule_extractor_agent.py # Prerequisite rule parser
│   ├── explanation_agent.py    # Course planner
│   └── verifier_agent.py       # Citation auditor
│
├── tools/
│   ├── vector_search_tool.py   # CrewAI tool: ChromaDB search
│   ├── pdf_tool.py             # CrewAI tool: page lookup
│   └── parser_tool.py          # CrewAI tool: prereq parsing
│
├── logic/
│   ├── transitive_reasoning.py # LLM-powered multi-hop prereq chains
│   ├── eligibility_checker.py  # High-level eligibility API
│   └── rule_engine.py          # Credit/grade rule extraction
│
├── crew/
│   ├── crew_setup.py           # Pipeline orchestration
│   └── tasks.py                # CrewAI task definitions
│
├── evaluation/
│   ├── evaluator.py            # 25-query eval runner
│   └── run_tests.py            # CLI evaluation entry point
│
├── app/
│   └── streamlit_app.py        # Full-featured chat UI
│
├── configs/
│   ├── config.yaml             # All configuration
│   └── model_config.py         # LLM + embedding factories
│
├── tests/
│   ├── test_rag.py             # Unit tests: loader, chunker, retriever
│   └── test_logic.py           # Unit tests: rules, transitive reasoning
│
├── main.py                     # Entry point (build/chat/eval/query)
└── requirements.txt
```

---

## ⚙️ RAG Pipeline Design

### Chunking Strategy
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 800 chars | Captures full course descriptions with prereqs (~200 tokens) |
| Overlap | 150 chars (18%) | Prevents prerequisite clauses from being split at boundaries |
| Splitter | RecursiveCharacterTextSplitter | Respects paragraph → sentence → word hierarchy |
| Separators | `\n\n`, `\n`, `. `, ` ` | Keeps logical course entries together |

### Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Why:** Fast (CPU-friendly), strong semantic similarity for English academic text, free

### Vector Store
- **ChromaDB** with cosine similarity
- Persistent to disk (`vectorstore/chroma_db/`)
- **Retriever:** `k=6`, `score_threshold=0.3` — balances recall vs. context length

### Transitive Prerequisite Reasoning
Rather than extracting a prerequisite graph (unreliable from OCR'd PDFs), the system uses
**LLM-powered transitive reasoning**:
1. Retrieve chunks for the target course AND its known prerequisites
2. Inject all evidence into a structured prompt
3. LLM reasons through A→B→C chains step-by-step with citations at each hop
4. This handles AND/OR logic, grade requirements, and implicit chains naturally

---

## 🤖 Agent Roles

| Agent | Role | Key Tools |
|-------|------|-----------|
| **Intake Agent** | Parse student query, identify missing info, ask ≤5 clarifying questions | — |
| **Retriever Agent** | Run targeted semantic queries over ChromaDB, return cited excerpts | `catalog_vector_search`, `catalog_page_lookup` |
| **Rule Extractor Agent** | Parse retrieved text → structured rules (AND/OR prereqs, grades, co-reqs, exceptions) | `prerequisite_parser`, `catalog_vector_search` |
| **Planner Agent** | Generate cited course plan or eligibility decision | — |
| **Verifier Agent** | Audit citations, flag hallucinations, output PASS/FAIL | — |

---

## 📤 Output Format

Every response follows this mandatory structure:

```
Answer / Plan:
[Eligibility decision or proposed course list]

Why (requirements/prereqs satisfied):
[Step-by-step reasoning with evidence]

Citations:
- kuk_prospectus_2011 | Page 42 | Chunk kuk_prospectus_2011_p42_c0
- kuk_prospectus_2011 | Page 87 | Chunk kuk_prospectus_2011_p87_c2

Clarifying questions (if needed):
[Questions if info is incomplete, or "None"]

Assumptions / Not in catalog:
[Information not verifiable from the catalog — student should verify with advisor]
```

---

## 📊 Evaluation

### Test Set (25 Queries)
| Category | Count | Description |
|----------|-------|-------------|
| Prerequisite checks | 10 | Direct eligible/not-eligible checks |
| Prerequisite chains | 5 | Multi-hop (A→B→C) reasoning |
| Program requirements | 5 | Credits, electives, policies |
| Not in docs / tricks | 5 | Must abstain (availability, faculty, schedules) |

### Metrics
- **Citation coverage rate** — % responses with ≥1 citation
- **Eligibility correctness** — % prereq decisions matching expected (manual rubric)
- **Abstention accuracy** — % of "not in docs" queries correctly refused
- **Avg latency** — seconds per query (Groq: typically <5s)

### Run Evaluation
```bash
python main.py --evaluate
# or
python evaluation/run_tests.py --max 10 --delay 3.0
```

Results saved to `data/evaluation/eval_results.json`.

---

## 📚 Data Sources

| Source | URL | Date Accessed | Coverage |
|--------|-----|---------------|----------|
| KUK Prospectus 2011 | Kurukshetra University official website | March 2026 | Course descriptions, prerequisites, program requirements, academic policies |

Additional public catalog data can be added by placing PDFs in `data/raw/` and re-running `--build-index`.

---

## 🛡️ Anti-Hallucination Controls

1. **Evidence-only generation prompt** — Every agent prompt explicitly states: "Use ONLY the retrieved catalog evidence"
2. **Citation required** — Verifier agent flags any claim without a citation
3. **Safe abstention** — System detects "not in catalog" queries and routes to abstain response
4. **Score threshold** — ChromaDB only returns chunks with similarity ≥ 0.3
5. **Verifier gate** — Final agent audits output before returning to user; can force REWRITE

---

## 🧪 Unit Tests

```bash
pytest tests/ -v
```

---

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *required* | Groq API key (free at console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_DB_PATH` | `vectorstore/chroma_db` | ChromaDB persistence path |
| `PDF_PATH` | `data/raw/kuk_prospectus_2011.pdf` | Path to catalog PDF |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVER_K` | `6` | Top-K chunks per query |

---

## 📝 Short Write-Up

### Architecture
5-agent sequential CrewAI pipeline: Intake → Retrieval → Rule Extraction → Planning → Verification. ChromaDB (cosine) for vector search, Groq llama-3.3-70b-versatile for all LLM calls (low latency, free tier).

### Chunking/Retrieval Choices
800-char chunks with 150-char overlap using RecursiveCharacterTextSplitter. The overlap is critical for academic catalogs where prerequisites often appear at the end of a course block and the beginning of the next. k=6 retrieves enough context for multi-hop chains without overflowing the context window.

### Transitive Reasoning
Instead of graph extraction (brittle with OCR'd PDFs), the system uses LLM reasoning over retrieved evidence. Multi-hop queries trigger multiple retrieval passes (target course + its prerequisites), and the LLM chains the reasoning with citations at every hop.

### Key Failure Modes
1. OCR errors in scanned PDFs can corrupt course names → mitigated by fuzzy keyword matching in retrieval
2. Co-requisite vs prerequisite confusion in catalog text → Rule Extractor agent specifically handles this distinction
3. Groq rate limits on rapid batch evaluation → configurable delay in evaluator

### Next Improvements
- Add re-ranking (cross-encoder) for better chunk selection
- Build explicit prerequisite graph for instant multi-hop lookups
- Add HyDE (Hypothetical Document Embeddings) for better retrieval on vague queries
- Cache frequent queries to reduce latency
