# 🎓 Agentic RAG Course Planning Assistant

An AI-powered academic assistant for **Kurukshetra University (KUK)** built using CrewAI, LangChain, ChromaDB, and Streamlit. This project addresses the "Agentic RAG Challenge" by helping students plan courses and answering prerequisite questions that are **strictly grounded** in program and course catalog documents.

## 🚀 Features

1. **Course Planning & Prerequisites:** Agents evaluate user requests, retrieve exact policy excerpts, and recommend course plans.
2. **Grounded with Citations:** Every claim is tied back to the source PDF, page number, and section.
3. **Agentic Strategy (CrewAI):**
    - **Intake Agent:** Normalizes user requests and asks clarifying questions if incomplete.
    - **Retriever Agent:** Searches the vector store for context.
    - **Rule Extractor Agent:** Extracts complex logic like transitive prerequisites and grade rules.
    - **Planner (Explanation) Agent:** Formats plans according to requirements and restrictions.
    - **Verifier/Auditor Agent:** Halts undocumented claims, corrects missing citations, and acts as the safety layer.
4. **Safe Abstention:** Gracefully refuses and suggests contacting an advisor if the catalog documents do not contain the answer.

## 📂 Project Structure

```
ai-course-planner-rag/
│
├── data/               # 📚 Place your raw PDFs (e.g. KUK Handbook) here
├── vectorstore/        # 🧠 ChromaDB persistence directory
├── rag/                # 🔥 Ingestion, chunking, and embedding logic
├── agents/             # 🤖 CrewAI Agents configuration
├── tools/              # 🛠️ Custom Search & Parsing Tools
├── logic/              # ⚙️ Core Reasoning Functions
├── crew/               # 🧠 Task Definitions & Workflows
├── evaluation/         # 📊 Automated grading & tests
├── app/                # 🌐 Streamlit UI
├── configs/            # ⚙️ Settings (model, RAG logic, API)
├── main.py             # 🚀 Entrypoint
└── requirements.txt    # 📦 Dependencies
```

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Linux/Mac
python -m pip install -r requirements.txt
```

### 2. Add Your Free API Key
1. Get a **Groq API Key** for free from [console.groq.com/keys](https://console.groq.com/keys) (We use Groq's `llama-3.3-70b-versatile` for high-quality, fast inference).
2. Rename `.env.example` -> `.env` and paste your key.
```ini
GROQ_API_KEY=gsk_your_key_here
```

### 3. Add Your Data
Create `data/raw/` and drop at least 25+ pages of academic catalogs in PDF format (e.g. `kuk_handbook_2025.pdf`) into it.

---

## ▶️ Usage

### Build the Database
Ingest, clean, chunk, and embed the PDFs locally (Free using `sentence-transformers`):
```bash
python main.py --build
```

### Run an Evaluation
Run the automated evaluation query set of 25 tricky queries covering prerequisites, course chains, program rules, and abstention edge-cases:
```bash
python main.py --eval
```

### Run a Single Manual Query via CLI
```bash
python main.py --query "Can I take CS301 next term if I've only completed CS101?"
```

### Launch the User Interface (Streamlit)
Chat with your new Assistant:
```bash
streamlit run app/streamlit_app.py
# or
python main.py --ui
```

---

## 🏗️ Architecture Tradeoffs

- **Chunking Strategy:** Chunk size is set to 1,000 chars with an overlapping 200 chars. This protects contextual bounds specifically related to lists (e.g. prerequisite chains A -> B -> C).
- **Embeddings Model:** `all-MiniLM-L6-v2` is used as a local model over OpenAI’s `text-embedding-3`. It uses CPU processing silently, ensuring the only API needed is for reasoning (Groq).
- **Retrieval:** Employs cosine similarity mapping, but we specifically separate retrieval tool instances (Planning Search vs Prerequisite Search vs General Policy).
