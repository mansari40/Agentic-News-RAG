from rag_baseline.generation.answer_generator import AnswerGenerator
from rag_baseline.retrieval.semantic_retriever import SemanticRetriever


class RAGPipeline:
    """
    Baseline Retrieval-Augmented Generation pipeline.

    Flow:
        question -> retrieve -> generate -> answer
    """

    def __init__(self) -> None:
        self.retriever = SemanticRetriever()
        self.generator = AnswerGenerator()

    def answer(self, question: str) -> str:
        """
        Run full RAG pipeline.

        Args:
            question: user query

        Returns:
            generated answer
        """
        chunks = self.retriever.retrieve(question)
        answer: str = self.generator.generate(question, chunks)
        return answer
