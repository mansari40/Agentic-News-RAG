from rag_baseline.domain_models import RetrievalResult
from rag_baseline.generation.answer_generator import AnswerGenerator
from rag_baseline.retrieval.semantic_retriever import SemanticRetriever

retriever = SemanticRetriever()
generator = AnswerGenerator()

print("=" * 60)
print("Timber Market Intelligence Assistant")
print("=" * 60)
print()
print("Hello! I'm your Timber Market Intelligence Assistant.")
print("Ask me anything about the timber market, recent news, or trends.")
print()
print("-" * 60)
print()

history: list[tuple[str, str]] = []
last_chunks: list[RetrievalResult] = []

while True:
    try:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
            print()
            print("Assistant: Goodbye! Have a great day!")
            print()
            break

        # sources from last answer
        if user_input.lower() == "sources":
            if last_chunks:
                print()
                print("Assistant: Here are the sources I used:")
                print()
                seen: set[str] = set()
                for i, chunk in enumerate(last_chunks, 1):
                    if chunk.article_id not in seen:
                        seen.add(chunk.article_id)
                        print(f"  [{i}] {chunk.article_id}")
                print()
            else:
                print()
                print("Assistant: No sources available yet. Ask a question first!")
                print()
            continue

        # Build query with conversation context
        if history:
            last_q, last_a = history[-1]
            full_query = (
                f"Previous question: {last_q}\n"
                f"Previous answer summary: {last_a[:200]}\n"
                f"New question: {user_input}"
            )
        else:
            full_query = user_input

        # Retrieve and generate
        last_chunks = retriever.retrieve(full_query)
        answer = generator.generate(full_query, last_chunks)

        print()
        print(f"Assistant: {answer}")
        print()
        print(f"  (Based on {len(last_chunks)} sources - type 'sources' to see them)")
        print()

        history.append((user_input, answer))

    except KeyboardInterrupt:
        print()
        print()
        print("Assistant: Session ended. Goodbye!")
        print()
        break
