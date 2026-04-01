"""
RAGAS evaluation script — Agentic RAG vs Baseline RAG
Metrics: faithfulness, answer_relevancy, context_precision, context_recall
"""

import csv
import os
from pathlib import Path
from typing import Any

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

_MAX_CTX_CHARS = 1200
_OUT_OF_SCOPE_MARKER = "outside the German timber market scope"

_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

for _m in [faithfulness, answer_relevancy, context_precision, context_recall]:
    _m.llm = _llm
    if hasattr(_m, "embeddings"):
        _m.embeddings = _emb

EVALS_DIR = Path(__file__).parent
DATASETS_DIR = EVALS_DIR / "datasets"
RESULTS_DIR = EVALS_DIR / "results"

FILES: dict[str, Any] = {
    "agentic": {
        "path": DATASETS_DIR / "agentic_rag_20 comparison questions.csv",
        "encoding": "latin-1",
        "context_cols": [f"context_{i}" for i in range(1, 9)],
    },
    "baseline": {
        "path": DATASETS_DIR / "baseline_rag_20 comparison questions.csv",
        "encoding": "latin-1",
        "context_cols": [f"context_{i}" for i in range(1, 6)],
    },
}

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def load_dataset(name: str) -> Dataset:
    cfg: dict[str, Any] = FILES[name]
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    skipped = 0
    with open(str(cfg["path"]), encoding=str(cfg["encoding"])) as f:
        for row in csv.DictReader(f):
            gt = row["ground_truth"].strip()
            if _OUT_OF_SCOPE_MARKER in gt:
                skipped += 1
                continue
            questions.append(row["Query"].strip())
            answers.append(row["Response"].strip())
            ctx = [
                row[c].strip()[:_MAX_CTX_CHARS]
                for c in cfg["context_cols"]
                if row.get(c, "").strip()
            ]
            contexts.append(ctx if ctx else [""])
            ground_truths.append(gt)

    if skipped:
        print(f"  Skipped {skipped} out-of-scope questions (evaluated separately)")

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def run_evaluation(name: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name.upper()}")
    print(f"{'='*60}")

    ds = load_dataset(name)
    print(f"  Loaded {len(ds)} rows")

    result = evaluate(ds, metrics=METRICS)

    df: pd.DataFrame = result.to_pandas()
    out_path = RESULTS_DIR / f"ragas_results_{name}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Per-question scores saved -> {out_path.name}")

    return df


def print_comparison(agentic_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n" + "=" * 62)
    print(f"  {'METRIC':<25} {'AGENTIC':>10} {'BASELINE':>10} {'DELTA':>10}")
    print("=" * 62)

    total_a, total_b, count = 0.0, 0.0, 0
    for m in metric_names:
        a = float(agentic_df[m].dropna().mean()) if m in agentic_df.columns else float("nan")
        b = float(baseline_df[m].dropna().mean()) if m in baseline_df.columns else float("nan")
        delta = a - b
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "=")
        print(f"  {m:<25} {a:>10.4f} {b:>10.4f} {arrow} {delta:>+.4f}")
        total_a += a
        total_b += b
        count += 1

    avg_a = total_a / count
    avg_b = total_b / count
    avg_delta = avg_a - avg_b
    arrow = "↑" if avg_delta > 0.01 else ("↓" if avg_delta < -0.01 else "=")
    print("-" * 62)
    print(f"  {'AVERAGE':<25} {avg_a:>10.4f} {avg_b:>10.4f} {arrow} {avg_delta:>+.4f}")
    print("=" * 62)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise OSError("OPENAI_API_KEY not set — check your .env file")

    agentic_result = run_evaluation("agentic")
    baseline_result = run_evaluation("baseline")

    print_comparison(agentic_result, baseline_result)
    print("\nDone. Full per-question breakdowns saved to evals/results/ragas_results_*.csv\n")
