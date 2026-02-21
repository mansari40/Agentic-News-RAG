import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag_baseline.orchestration.rag_pipeline import RAGPipeline


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize resources on startup, cleanup on shutdown"""
    print("Initializing Baseline RAG pipeline...")
    app.state.rag_pipeline = RAGPipeline(use_hybrid=True)
    print("Baseline RAG ready")
    yield
    print("Shutting down...")


app: FastAPI = FastAPI(
    title="Timber Market Intelligence API",
    description="REST API for Baseline RAG system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    use_hybrid: bool = True
    top_k: int = Field(5, ge=1, le=10)


class SourceResponse(BaseModel):
    source: str | None = None
    title: str | None = None
    url: str | None = None
    content: str
    score: float
    keywords: list[str] = []
    published_at: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    response_time: float
    search_type: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


@app.get("/", response_model=dict[str, str])  # type: ignore[misc]
def root() -> dict[str, str]:
    return {
        "message": "Timber Market Intelligence API - Baseline RAG",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)  # type: ignore[misc]
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
    )


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)  # type: ignore[misc]
async def query_baseline_rag(request: QueryRequest) -> QueryResponse:
    start_time = time.time()

    try:
        rag: RAGPipeline = app.state.rag_pipeline
        rag.use_hybrid = request.use_hybrid

        sources = rag.retriever.retrieve(
            request.question,
            use_hybrid=request.use_hybrid,
        )

        answer: str = rag.generator.generate(
            request.question,
            sources[: request.top_k],
        )

        response_time = time.time() - start_time

        formatted_sources: list[SourceResponse] = []
        for source in sources[: request.top_k]:
            formatted_sources.append(
                SourceResponse(
                    source=source.source,
                    title=source.title,
                    url=source.url,
                    content=source.content[:400] + "..."
                    if len(source.content) > 400
                    else source.content,
                    score=source.similarity_score,
                    keywords=source.keywords[:5] if source.keywords else [],
                    published_at=str(source.published_at) if source.published_at else None,
                )
            )

        return QueryResponse(
            answer=answer,
            sources=formatted_sources,
            response_time=round(response_time, 2),
            search_type="hybrid" if request.use_hybrid else "vector",
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
