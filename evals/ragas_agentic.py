"""
RAGAS evaluation script - Agentic RAG 55-question full evaluation
Metrics: faithfulness, answer_relevancy, context_precision, context_recall
Out-of-scope questions (6) are excluded from metrics and reported separately.
requires_live_search questions (8) are included in RAGAS evaluation.
"""

import csv
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset  # type: ignore[attr-defined]
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

load_dotenv()

_MAX_CTX_CHARS = 1500
_OUT_OF_SCOPE_MARKER = "outside the German timber market scope"
_LIVE_SEARCH_MARKER = "[requires_live_search]"

EVALS_DIR = Path(__file__).parent
DATASETS_DIR = EVALS_DIR / "datasets"
RESULTS_DIR = EVALS_DIR / "results"

INPUT_FILE = DATASETS_DIR / "agentic_rag_55 questions.csv"
OUTPUT_FILE = RESULTS_DIR / "ragas_results_agentic_55.csv"
CONTEXT_COLS = [f"context_{i}" for i in range(1, 9)]

_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=120))
_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
for _m in METRICS:
    _m.llm = _llm
    if hasattr(_m, "embeddings"):
        _m.embeddings = _emb


def load_dataset() -> tuple[Dataset, int, int]:
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    skipped_oos = 0
    skipped_live = 0

    with open(str(INPUT_FILE), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gt = row["ground_truth"].strip()
            qtype = row.get("query_type", "").strip().lower()

            # Exclude out-of-scope queries (by explicit query_type flag or ground-truth marker)
            if qtype == "out_of_scope" or _OUT_OF_SCOPE_MARKER in gt:
                skipped_oos += 1
                continue

            # Include requires_live_search queries in main metrics
            if qtype == "requires_live_search":
                skipped_live += 1

            questions.append(row["Query"].strip())
            answers.append(row["Response"].strip())
            ctx = [row[c].strip()[:_MAX_CTX_CHARS] for c in CONTEXT_COLS if row.get(c, "").strip()]
            contexts.append(ctx if ctx else [""])
            ground_truths.append(gt.replace(_LIVE_SEARCH_MARKER, "").strip())

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    return ds, skipped_oos, skipped_live


def print_results(df: pd.DataFrame) -> None:
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n" + "=" * 52)
    print(f"  {'METRIC':<25} {'SCORE':>10} {'N':>6}")
    print("=" * 52)

    total, count = 0.0, 0
    for m in metric_names:
        if m not in df.columns:
            print(f"  {m:<25} {'N/A':>10}")
            continue
        valid = df[m].dropna()
        score = float(valid.mean())
        n = len(valid)
        print(f"  {m:<25} {score:>10.4f} {n:>6}")
        total += score
        count += 1

    avg = total / count if count else 0.0
    print("-" * 52)
    print(f"  {'AVERAGE':<25} {avg:>10.4f}")
    print("=" * 52)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise OSError("OPENAI_API_KEY not set — check your .env file")

    print(f"\n{'='*52}")
    print("  Agentic RAG — 55 Question Evaluation")
    print(f"{'='*52}")

    ds, skipped_oos, skipped_live = load_dataset()
    print(f"  Loaded {len(ds)} questions for RAGAS evaluation")
    print(f"  Skipped {skipped_oos} out-of-scope questions (scope detection evaluated separately)")
    print(f"  Included {skipped_live} requires-live-search questions (evaluated within RAGAS)")

    result = evaluate(ds, metrics=METRICS)

    df: pd.DataFrame = result.to_pandas()
    df.to_csv(str(OUTPUT_FILE), index=False, encoding="utf-8")
    print(f"  Per-question scores saved -> {OUTPUT_FILE.name}")

    print_results(df)
    print("\nDone.\n")
