<div align="center">

# Timber Intel - Agentic RAG for German Timber Intelligence Market

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14%2B-black)](https://nextjs.org/)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-blue)](https://timber-intel.onrender.com/)
[![GitHub](https://img.shields.io/badge/GitHub-mansari40-lightgrey)](https://github.com/mansari40/Agentic-News-RAG)

<p>If you find my project helpful, please give me a star ⭐ on GitHub!</p>

**A production-oriented retrieval-augmented generation platform for German timber market intelligence, comparing fast baseline retrieval with agentic reasoning, tool orchestration, and evidence validation.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key capabilities](#key-capabilities)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Baseline vs Agentic](#baseline-vs-agentic)
  - [Baseline RAG](#baseline-rag)
  - [Agentic RAG](#agentic-rag)
  - [User Interface](#user-interface)
- [Backend API](#backend-api)
- [Evaluation and research](#evaluation-and-research)
- [Setup](#setup)
- [Run locally](#run-locally)
- [Data ingestion and indexing](#data-ingestion-and-indexing)
- [Deployment](#deployment)
- [Development](#development)

## Overview

This is my master thesis project: a dual-path RAG architecture designed for German timber market intelligence:

- **Baseline path:** fast, grounded semantic retrieval with transparent source handling
- **Agentic path:** tool-augmented research with dynamic planning, evidence validation, and traceable reasoning

## Key capabilities

- Dual-mode query experience: `Baseline` vs `Agentic`
- Evidence-backed answer generation with confidence metadata
- Qdrant vector search plus PostgreSQL metadata storage
- Streaming Server-Sent Events for live agentic trace updates
- Conversation memory, query caching, and follow-up support in the agentic path
- Evaluation tooling using RAGAS and LLM-based judgment

## Architecture

```
Timber Intel (Agentic News RAG)
├─ Data Layer/
│  ├─ MediaStack API          # Live news ingestion
│  ├─ PostgreSQL             # Article metadata & persistence
│  └─ Qdrant Cloud           # Vector embeddings & semantic search
│
├─ Retrieval Pipelines/
│  ├─ Baseline RAG
│  │  ├─ Embeddings (text-embedding-3-small)
│  │  ├─ Hybrid search (vector + keyword)
│  │  └─ Source ranking & synthesis
│  │
│  └─ Agentic RAG
│     ├─ Planner (query classification)
│     ├─ Researcher (tool orchestration & ReAct loop)
│     ├─ Ranker (candidate filtering)
│     ├─ Verifier (evidence validation)
│     └─ Synthesizer (answer generation)
│
├─ Tools/
│  ├─ search_tavily_specialist
│  ├─ search_tavily_web
│  ├─ search_mediastack
│  └─ search_baseline
│
├─ Evaluation/
│  ├─ `evals/ragas_agentic.py`        # 55-question RAGAS evaluation
│  ├─ `evals/ragas_comparison.py`     # agentic vs baseline comparison
│  ├─ `evals/llm_judge.ipynb`         # notebook judgment analysis
│  └─ `evals/datasets/`               # curated evaluation data
│
└─ Delivery Layer/
   ├─ backend/ (FastAPI + SSE streaming)
   └─ frontend/ (Next.js dashboard)
```

## Repository layout

- `backend/` — FastAPI app, query routes, stream endpoint, memory and cost APIs
- `frontend/` — Next.js UI with chat, comparison view, analytics, and architecture panels
- `src/agentic_rag/` — agentic pipeline, ReAct workflow, memory, prompts, and tool integration
- `src/rag_baseline/` — baseline embedding, retrieval, and generation code
- `data/` — article CSV datasets, persistence directories for PostgreSQL and Qdrant
- `evals/` — evaluation scripts, notebooks, datasets, and result outputs
- `scripts/` — ingestion and indexing entrypoints
- `tests/` — pytest coverage for pipeline components
- `htmlcov/` — coverage report artifacts
- `logs/` — runtime logs and diagnostics

## Baseline vs Agentic

### Baseline RAG

The baseline path uses `RAGPipeline` for fast semantic retrieval and answer generation.

![Baseline RAG workflow](./assets/screenshots/baseline_rag.png)

How the baseline RAG works:

- Converts MediaStack 1263 unique article to 1,385 chunks and turns them into embeddings and stores them in Qdrant for semantic similarity search
- Supports both vector-based and hybrid retrieval to balance semantic match quality with metadata signals
- Filters top 5 documents by relevance score and passes evidence directly to the answer generator
- Returns source metadata, provenance, and snippets so users can inspect grounding
- Optimized for low-latency, deterministic lookup behavior on fact-based queries.

### Agentic RAG

The agentic path uses a ReAct-style workflow with tool calls and source verification.

![Agentic RAG architecture](./assets/screenshots/agentic_rag.png)

How the agentic RAG works:

- The planner classifies the question and selects a research strategy, including tool usage and query decomposition
- The researcher invokes tavily specialist search tools,tavily web search and MediaStack articles then  merges candidate evidence, and tracks observations in a scratchpad
- The ranker filters and prioritizes evidence, while the verifier checks relevance and factual consistency
- The synthesizer assembles the final answer only from validated sources and produces confidence metadata
- The streaming endpoint emits workflow steps, researcher actions, and reasoning events in real time
- Memory, query caching, and conversational fallback ensure repeatability and better follow-up handling

### User Interface

![Timber Intel UI](./assets/screenshots/ui.png)

Short Summary of UI:

- The deployed interface supports a short analytical workflow (not just single-turn chat): users can submit and manage multiple queries, retrieve and tune answer configurations, and inspect supporting evidence from a single continuous workspace.
- Answers present a headline finding plus detailed explanation, followed by a metadata bar (processing time, number of sources, query cost, and token usage) and a concise evidence statement.
- Each answer view exposes a `Key facts` box, a collapsible reasoning trace with the full researcher scratchpad, and an expandable list of all sources so users can inspect provenance and selection rationale.
- Multiple queries can run in parallel; prior interactions are stored locally in the browser and accessible from the sidebar for quick reference.
- Enabling Research Mode turns on session-level query logging; session metadata (costs, latency, tools used, query distribution) can be exported from the Analytics tab after a session.
- Additional UI tabs: A/B Test (compare baseline vs agentic answers), Architecture (embedded pipeline explanation), Analytics (session summaries and interactive graphs), and Overview (usage guide and operational bounds).

## Backend API

The FastAPI backend exposes these routes:

- `GET /` — service metadata
- `GET /health` — pipeline health status
- `GET /coverage` — agentic coverage diagnostics
- `POST /query/baseline` — baseline RAG query
- `POST /query/agentic` — agentic RAG query
- `POST /query/agentic/stream` — streaming agentic trace (SSE)
- `POST /agentic/memory/clear` — clear agentic session memory
- `GET /agentic/memory/summary` — conversation memory summary
- `GET /agentic/cost` — session cost metadata

## Evaluation and research

This repository includes dedicated evaluation work for automated RAG metrics and human-like LLM judgment.

- `evals/ragas_agentic.py` — runs RAGAS evaluation on a curated 55-question dataset, excluding out-of-scope queries and including live-search cases.
- `evals/ragas_comparison.py` — compares agentic and baseline outputs using 20 paired comparison queries and reports per-metric deltas.
- `evals/llm_judge.ipynb` — notebook-based LLM judgment workflows for comparative quality assessment.
- `evals/ragas_analysis.ipynb` — exploratory analysis and visualization of evaluation outcomes.
- `evals/datasets/` — curated evaluation datasets and query collections.
- `evals/results/` — persisted evaluation outputs and CSV reports.

Evaluation focus:

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

Key notes:

- The agentic evaluation script loads 55 questions and skips out-of-scope queries while keeping live-search cases in the main metric set.
- The comparison script generates separate baseline and agentic score files and prints side-by-side deltas for the same query pairs.

Run evaluation:

```bash
cd evals
python ragas_agentic.py
python ragas_comparison.py
```

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- Docker + Docker Compose

### Python setup

```powershell
cd Agentic-News-RAG
python -m venv .venv
.\.venv\Scripts\activate
pip install -e "[dev]"
```

### Frontend setup

```bash
cd frontend
npm install
```

### Environment configuration

Copy the example configuration and populate the secrets:

```powershell
copy .env.example .env
```

Required values:

- `OPENAI_API_KEY`
- `MEDIASTACK_API_KEY`
- `TAVILY_API_KEY`
- `DB_PASSWORD`
- `DATABASE_URL`
- `LANGCHAIN_API_KEY`

## Run locally

### Start infrastructure

```bash
docker compose up -d
```

### Launch backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Launch frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` to use the dashboard.

## Data ingestion and indexing

```bash
python scripts/run_ingestion.py
python scripts/run_indexing.py
```

- `run_ingestion.py` ingests timber news and creates article chunks
- `run_indexing.py` builds the Qdrant vector index for retrieval

## Deployment

This project is deployed across multiple managed platforms:

- **Application:** [Render](https://render.com/) (FastAPI backend + Next.js frontend)
- **Vector store:** [Qdrant Cloud](https://cloud.qdrant.io/) (managed vector database)
- **Database:** [Neon](https://neon.tech/) (PostgreSQL on serverless infrastructure)

All services are configured for high availability, auto-scaling, and zero-downtime deployments.


## Development

Run the test suite:

```bash
pytest
```

Run formatting and linting:

```bash
python -m pre_commit run -a
```

---

<div align="center">
  Made with ❤️ by Mustafa
</div>
