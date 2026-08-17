# MultiAgent-RAG

A multi-agent Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, FAISS, and Groq-hosted LLMs. Instead of a single "retrieve-then-generate" pass, incoming questions are routed, rewritten, decomposed, retrieved with a hybrid dense+sparse search, answered, and then self-critiqued — with a Streamlit chat UI exposing the full reasoning trace.

The included example corpus is a healthcare-AI article collection (`healthcare_ai_corpus_v2.pdf`), but the pipeline is domain-agnostic and can be pointed at any structured document corpus.

## How it works

Each query flows through a chain of small, focused agents orchestrated by `Orchestrator.process_query()`:

1. **Router** — classifies the query as `factual`, `follow_up`, or `out_of_scope` using conversation history.
2. **Query Rewriter** — if the query is a follow-up, rewrites it into a standalone question using chat history.
3. **Decomposer** — breaks complex questions into smaller sub-queries for more targeted retrieval.
4. **Hybrid Retrieval** — for each sub-query, runs dense (FAISS) and sparse (BM25) search in parallel, fuses results with Reciprocal Rank Fusion, and reranks the fused pool with a cross-encoder (`ms-marco-MiniLM-L-6-v2`).
5. **Reasoner** — generates a grounded answer with inline citations from the retrieved context, enforcing numeric/percentage accuracy.
6. **Critic** — independently evaluates the answer against the retrieved context and flags issues.
7. **Memory** — the query/answer pair is stored in a rolling chat history used for follow-up resolution.

The API returns both the final answer and a full trace (route, rewritten query, sub-queries, citations, reasoning, critic verdict), which the Streamlit UI renders as an expandable "reasoning pipeline" panel.

## Project structure

```
rag-agent/
├── app/
│   ├── agents/          # router, query_rewriter, decomposer, reasoner, critic, orchestrator
│   ├── api/              # FastAPI app, routes, request/response schemas
│   ├── ingestion/        # PDF/HTML loading, chunking, embedding, FAISS indexing
│   ├── memory/           # rolling chat history
│   ├── models/           # Groq LLM client wrapper
│   ├── prompts/           # prompt templates for each agent
│   ├── retrieval/        # dense/sparse retriever, RRF fusion, cross-encoder reranker
│   ├── utils/             # logging, helpers
│   └── constants.py
├── data/
│   ├── raw/               # source PDFs (corpus + eval set)
│   └── index/              # generated FAISS index + metadata
├── eval/
│   ├── questions.json      # eval question set
│   ├── eval_runner.py       # runs the pipeline over the eval set
│   └── scorer.py            # LLM-as-judge scoring
├── scripts/
│   ├── build_index.py        # ingestion → chunking → embedding → indexing
│   └── test_retrieval.py
└── ui/
    └── streamlit_app.py       # chat UI with reasoning trace viewer
```

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| LLM | Groq (`openai/gpt-oss-20b` for agents, `openai/gpt-oss-120b` as judge) |
| Embeddings | `sentence-transformers` (`BAAI/bge-large-en`) |
| Vector index | FAISS (flat L2) |
| Keyword search | `rank-bm25` |
| Reranking | `sentence-transformers` cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| PDF parsing | PyMuPDF (`fitz`), `pypdf` |
| Web scraping | `requests` + `BeautifulSoup` |
| Dependency management | `uv` |

## Setup

**Requirements:** Python ≥ 3.13, a [Groq API key](https://console.groq.com/), and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/prithviraj-0710/MultiAgent-RAG.git
cd MultiAgent-RAG/rag-agent

# install dependencies
uv sync
```

Create a `.env` file in `rag-agent/`:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Building the index

The FAISS index and BM25 corpus are built from the source PDF in `data/raw/`. The loader parses article entries out of the corpus PDF, fetches each article's linked source page, chunks the text, embeds it, and writes the index to `data/index/`.

```bash
uv run python scripts/build_index.py
```

This produces `data/index/faiss.index` and `data/index/metadata.pkl`.

## Running the app

Start the API:

```bash
uv run uvicorn app.api.routes:app --reload
```

In a separate terminal, start the UI:

```bash
uv run streamlit run ui/streamlit_app.py
```

The UI expects the API at `http://127.0.0.1:8000`.

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/query` | Runs a query through the full agent pipeline. Body: `{"query": "..."}` |
| `POST` | `/reset` | Clears the server-side chat memory |

Example:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key risks of AI in clinical diagnostics?"}'
```

## Evaluation

`eval/questions.json` contains a fixed question set with expected answers and sources. Running the evaluator replays every question through a fresh `Orchestrator`, scores each answer with an LLM judge, and writes results to `eval/results.json`.

```bash
uv run python eval/eval_runner.py
```

## Notes

- Retrieval alpha (dense vs. sparse weighting) and top-k are configurable in `hybrid_search()` calls within the orchestrator.
- Chat memory is in-process and capped at the last 5 exchanges; it resets when the API restarts or `/reset` is called.
- The PDF loader includes some parsing logic (article-number regex, a hardcoded URL fix) tailored to the format of the bundled corpus — swap in your own loader logic for a different document source.
