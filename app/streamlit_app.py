import time
from typing import Any

import streamlit as st
from rag_baseline.domain_models import RetrievalResult
from rag_baseline.indexing.vector_indexer import VectorIndexingPipeline
from rag_baseline.ingestion.article_ingestor import IngestionService
from rag_baseline.orchestration.rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="Timber Market Intelligence",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingestion_complete" not in st.session_state:
    st.session_state.ingestion_complete = False


@st.cache_resource  # type: ignore[misc]
def get_rag_pipeline() -> RAGPipeline:
    """Initialize RAG pipeline (cached)"""
    return RAGPipeline(use_hybrid=True)


def run_ingestion_and_indexing() -> bool:
    """Run full ingestion and indexing pipeline"""
    try:
        # Ingestion
        with st.spinner("Ingesting articles from MediaStack..."):
            ingestor = IngestionService()
            ingestor.ingest()
            st.success("✓ Articles ingested successfully")

        # Indexing
        with st.spinner("Indexing chunks into vector database..."):
            indexer = VectorIndexingPipeline()
            indexer.index_all_chunks()
            st.success("✓ Vector indexing complete")

        st.session_state.ingestion_complete = True
        st.balloons()
        return True

    except Exception as e:
        st.error(f"Error during ingestion/indexing: {str(e)}")
        return False


def format_source(result: RetrievalResult, index: int) -> None:
    """Format a source as an expandable card"""
    with st.expander(
        f"Source {index + 1} - {result.source} (Relevance: {result.similarity_score:.2f})"
    ):
        if result.title:
            st.write(f"**Title:** {result.title}")
        if result.published_at:
            st.write(f"**Published:** {result.published_at}")
        if result.url:
            st.write(f"**Link:** [{result.url}]({result.url})")
        if result.keywords:
            keywords = ", ".join(result.keywords[:5])
            st.write(f"**Keywords:** {keywords}")

        st.write("**Content Preview:**")
        preview = result.content[:400] + "..." if len(result.content) > 400 else result.content
        st.text(preview)


def main() -> None:
    # Header
    st.title("Timber Market Intelligence System")
    st.markdown("Advanced RAG system for timber market analysis and insights")

    # Sidebar - System Controls
    with st.sidebar:
        st.header("System Controls")

        # Ingestion & Indexing
        st.subheader("Data Management")

        if st.button("Run Ingestion & Indexing", type="primary", use_container_width=True):
            run_ingestion_and_indexing()

        st.caption("Fetches latest articles and indexes them for search")

        st.divider()

        # Search Settings
        st.subheader("Search Configuration")

        use_hybrid = st.checkbox(
            "Enable Hybrid Search", value=True, help="Combines semantic and keyword-based search"
        )

        top_k = st.slider(
            "Number of Sources",
            min_value=3,
            max_value=10,
            value=5,
            help="How many sources to retrieve per query",
        )

        st.divider()

        # Session Stats
        st.subheader("Session Statistics")
        user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
        st.metric("Total Queries", len(user_messages))

        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main Chat Interface
    st.header("Ask Questions")

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources for assistant responses
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                st.divider()
                st.subheader("Sources")
                for idx, source in enumerate(message["sources"]):
                    format_source(source, idx)

                # Response metadata
                if "response_time" in message:
                    hybrid_text = "Hybrid" if message.get("hybrid", False) else "Vector Only"
                    st.caption(
                        f"Response time: {message['response_time']:.2f}s | "
                        f"Sources: {len(message['sources'])} | "
                        f"Search: {hybrid_text}"
                    )

    # Chat input
    if prompt := st.chat_input("Enter your question about timber markets..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"), st.spinner("Searching knowledge base..."):
            start_time = time.time()

            try:
                # Get RAG pipeline
                rag = get_rag_pipeline()
                rag.use_hybrid = use_hybrid

                # Retrieve sources
                sources = rag.retriever.retrieve(prompt, use_hybrid=use_hybrid)

                # Generate answer
                answer = rag.generator.generate(prompt, sources[:top_k])

                response_time = time.time() - start_time

                # Display answer
                st.markdown(answer)

                # Display sources
                if sources:
                    st.divider()
                    st.subheader("Sources")
                    for idx, source in enumerate(sources[:top_k]):
                        format_source(source, idx)

                    # Metadata
                    hybrid_text = "Hybrid" if use_hybrid else "Vector Only"
                    st.caption(
                        f"Response time: {response_time:.2f}s | "
                        f"Sources: {len(sources[:top_k])} | "
                        f"Search: {hybrid_text}"
                    )

                # Save to history
                message_data: dict[str, Any] = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources[:top_k],
                    "response_time": response_time,
                    "hybrid": use_hybrid,
                }
                st.session_state.messages.append(message_data)

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                error_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": error_msg,
                }
                st.session_state.messages.append(error_message)


if __name__ == "__main__":
    main()
