# 📄 PDF-Synapse — Chat With Your PDF using RAG + Gemini

A lightweight Retrieval-Augmented Generation (RAG) pipeline that lets you **ask questions about any PDF** and get answers grounded in its actual content — powered by **Qdrant** (vector store), **FastEmbed** (local embeddings), and **Google Gemini** (LLM).

No OpenAI key needed. No heavy GPU needed. Just a PDF, a Qdrant instance, and a Gemini API key.

---

## 🧠 How It Works

This project is split into two independent scripts — **ingest** and **query** — following the standard RAG pattern:

```
┌─────────────┐     chunk      ┌──────────────┐     embed      ┌─────────────┐
│   PDF File  │ ─────────────► │ Text Splitter│ ─────────────► │  FastEmbed  │
└─────────────┘                └──────────────┘                └──────┬──────┘
                                                                        │
                                                                        ▼
                                                                 ┌─────────────┐
                                                                 │   Qdrant    │
                                                                 │ Vector Store│
                                                                 └──────┬──────┘
                                                                        │
User Question ──────────────────────────────► Retriever (top-k) ◄─────┘
                                                        │
                                                        ▼
                                              ┌───────────────────┐
                                              │  Gemini (LLM)     │
                                              │ + Retrieved Chunks│
                                              └─────────┬─────────┘
                                                         ▼
                                                  Final Answer 🎯
```

### 1️⃣ Ingestion (`ingest.py`)
- Loads a PDF via `PyPDFLoader`
- Splits it into overlapping chunks (`chunk_size=1000`, `overlap=200`) using `RecursiveCharacterTextSplitter`
- Embeds each chunk locally using **FastEmbed** (`BAAI/bge-small-en-v1.5`) — no external embedding API calls, fast and free
- Pushes the vectors into a **Qdrant Cloud** collection (`pdf-synapse`)

### 2️⃣ Query (`query.py`)
- Connects to the same Qdrant collection
- Retrieves the top-3 most relevant chunks for a user's question
- Feeds them into **Gemini** as context via a LangChain `create_retrieval_chain`
- Prints a grounded answer — falling back to the model's own knowledge if the PDF doesn't contain the answer

---

## 🛠️ Tech Stack

| Component        | Tool                                       |
|-------------------|---------------------------------------------|
| PDF Parsing       | `langchain_community.PyPDFLoader`           |
| Chunking          | `RecursiveCharacterTextSplitter`            |
| Embeddings        | `FastEmbedEmbeddings` (`BAAI/bge-small-en-v1.5`) |
| Vector Store      | `Qdrant Cloud`                              |
| LLM               | `Gemini` via `langchain_google_genai`       |
| Orchestration     | `langchain-classic` retrieval chains        |

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd pdf-synapse
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

<details>
<summary><strong>requirements.txt (suggested)</strong></summary>

```
python-dotenv
langchain-community
langchain-text-splitters
langchain-google-genai
langchain-qdrant
langchain-classic
langchain-core
qdrant-client
fastembed
pypdf
```
</details>

---

## 🔑 Environment Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

- Get a **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey)
- Get a **free Qdrant Cloud cluster** from [cloud.qdrant.io](https://cloud.qdrant.io)

---

## 🚀 Usage

### Step 1 — Ingest your PDF
Place your PDF next to `ingest.py` and update the filename:

```python
pdf_path = Path(__file__).parent / "your-file.pdf"
```

Then run:

```bash
python ingest.py
```

This chunks, embeds, and uploads your PDF to Qdrant. ⚠️ Uses `force_recreate=True`, meaning **each run wipes and rebuilds the collection** — fine for single-doc use, but don't run it if you've added multiple PDFs you want to keep.

### Step 2 — Ask questions
```bash
python query.py
```

```
enter your question here - What is this document about?

Asking Gemini: 'What is this document about?'...

--- GEMINI'S ANSWER ---
<grounded answer based on retrieved chunks>
```

---

## ⚙️ Configuration Knobs

| Parameter                | Location    | Effect                                      |
|---------------------------|-------------|----------------------------------------------|
| `chunk_size` / `chunk_overlap` | `ingest.py` | Controls granularity of retrieved context |
| `search_kwargs={"k": 3}`  | `query.py`  | Number of chunks retrieved per question     |
| `temperature=0.3`         | `query.py`  | Lower = more deterministic answers          |
| `model_name` (FastEmbed)  | both files  | Must match between ingest & query           |

---

## 🐞 Known Issues / TODO

- [ ] `GoogleGenerativeAIEmbeddings` is imported in `ingest.py` but never used — dead import, safe to remove (embeddings are done via FastEmbed).
- [ ] Verify the Gemini model string (`gemini-3.5-flash`) against Google's current model list — model names change over time.
- [ ] `pdf_path` is hardcoded — consider accepting a filename via CLI arg (`sys.argv`) or `argparse` for multi-PDF support.
- [ ] `force_recreate=True` will delete existing vectors on every ingest run — swap to `force_recreate=False` if you want to accumulate multiple documents in one collection.
- [ ] No source citation in the answer (which page/chunk it came from) — could return `response["context"]` alongside the answer for transparency.
- [ ] No conversational memory — every question is stateless (no chat history passed to the chain).

---

## 🌱 Possible Next Steps

- Add a Streamlit/Gradio UI for a proper chat interface
- Support multi-PDF ingestion with metadata filtering (filename, page)
- Add conversational memory (`create_history_aware_retriever`)
- Stream responses token-by-token instead of waiting for full output
- Add citations/source highlighting in the answer

---

## 📜 License

MIT — do whatever you want with it.