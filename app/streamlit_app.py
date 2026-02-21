from typing import Any

import requests
import streamlit as st
from rag_baseline.indexing.vector_indexer import VectorIndexingPipeline
from rag_baseline.ingestion.article_ingestor import IngestionService

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

# API Configuration
API_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


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


def format_source(source: dict[str, Any], index: int) -> None:
    """Format a source as an expandable card"""
    with st.expander(f"Source {index + 1} - {source['source']} (Relevance: {source['score']:.2f})"):
        if source.get("title"):
            st.write(f"**Title:** {source['title']}")
        if source.get("url"):
            st.write(f"**Link:** [{source['url']}]({source['url']})")
        if source.get("keywords"):
            keywords = ", ".join(source["keywords"][:5])
            st.write(f"**Keywords:** {keywords}")

        st.write("**Content Preview:**")
        st.text(source["content"])


def main() -> None:
    # Header
    st.title("Timber Market Intelligence System")
    st.markdown("Advanced RAG system for timber market analysis and insights")

    # Check API health
    if not check_api_health():
        st.error(
            "Backend API is not running. Please start FastAPI server:\n"
            "```bash\n"
            "uvicorn backend.main:app --reload\n"
            "```"
        )
        st.stop()

    # Sidebar - System Controls
    with st.sidebar:
        st.header("System Controls")

        st.success("API Connected")

        st.divider()

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

        # Generate response via API
        with st.chat_message("assistant"), st.spinner("Searching knowledge base..."):
            try:
                # Call FastAPI backend
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "use_hybrid": use_hybrid, "top_k": top_k},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    response_time = data["response_time"]

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    if sources:
                        st.divider()
                        st.subheader("Sources")
                        for idx, source in enumerate(sources):
                            format_source(source, idx)

                        # Metadata
                        hybrid_text = "Hybrid" if use_hybrid else "Vector Only"
                        st.caption(
                            f"Response time: {response_time:.2f}s | "
                            f"Sources: {len(sources)} | "
                            f"Search: {hybrid_text}"
                        )

                    # Save to history
                    message_data: dict[str, Any] = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "response_time": response_time,
                        "hybrid": use_hybrid,
                    }
                    st.session_state.messages.append(message_data)
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.Timeout:
                error_msg = "Request timed out. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

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
