from agentic_rag.memory import ConversationMemory


class TestConversationMemory:
    """Test suite for ConversationMemory"""

    def test_memory_initialization(self) -> None:
        """Test memory can be initialized with custom max_history"""
        memory = ConversationMemory(max_history=5)
        assert memory.max_history == 5
        assert len(memory.conversation_history) == 0

    def test_add_interaction(self) -> None:
        """Test adding an interaction stores it in history"""
        memory = ConversationMemory()

        memory.add_interaction(
            query="What are lumber prices?",
            answer="Lumber prices are €500/m³",
            sources=[{"url": "https://timber-online.net"}],
            metadata={"entities": ["Lumber"]},
        )

        assert len(memory.conversation_history) == 1
        assert memory.conversation_history[0]["query"] == "What are lumber prices?"

    def test_max_history_limit(self) -> None:
        """Test history is capped at max_history (FIFO eviction)"""
        memory = ConversationMemory(max_history=3)

        for i in range(5):
            memory.add_interaction(query=f"Query {i}", answer=f"Answer {i}", sources=[])

        assert len(memory.conversation_history) == 3
        assert memory.conversation_history[0]["query"] == "Query 2"

    def test_query_cache(self) -> None:
        """Test exact query is cached after add_interaction"""
        memory = ConversationMemory()

        memory.add_interaction(query="lumber prices", answer="€500/m³", sources=[])

        cached = memory.check_query_cache("lumber prices")

        assert cached is not None
        assert cached["answer"] == "€500/m³"

    def test_cache_case_insensitive(self) -> None:
        """Test cache lookup is case-insensitive"""
        memory = ConversationMemory()

        memory.add_interaction(query="Lumber Prices", answer="€500/m³", sources=[])

        cached = memory.check_query_cache("lumber prices")

        assert cached is not None

    def test_temporal_queries_bypass_cache(self) -> None:
        """Test temporal queries are not cached"""
        memory = ConversationMemory()

        memory.add_interaction(query="latest lumber news today", answer="some news", sources=[])

        cached = memory.check_query_cache("latest lumber news today")

        assert cached is None

    def test_follow_up_detection_with_pronoun(self) -> None:
        """Test follow-up detection returns True for pronoun-based short questions"""
        memory = ConversationMemory()

        memory.add_interaction(query="What are lumber prices?", answer="€500/m³", sources=[])

        is_follow_up, context = memory.is_follow_up_question("What about it?")

        assert is_follow_up is True
        assert context is not None

    def test_follow_up_detection_negative(self) -> None:
        """Test follow-up detection returns False for unrelated queries"""
        memory = ConversationMemory()

        memory.add_interaction(query="What are lumber prices?", answer="€500/m³", sources=[])

        is_follow_up, context = memory.is_follow_up_question("What is the weather today in Berlin?")

        assert is_follow_up is False

    def test_follow_up_returns_false_with_no_history(self) -> None:
        """Test follow-up detection returns False when memory is empty"""
        memory = ConversationMemory()

        is_follow_up, context = memory.is_follow_up_question("Tell me more")

        assert is_follow_up is False
        assert context is None

    def test_conversation_context(self) -> None:
        """Test get_conversation_context returns recent Q&A pairs"""
        memory = ConversationMemory()

        memory.add_interaction(query="Query 1", answer="Answer 1", sources=[])
        memory.add_interaction(query="Query 2", answer="Answer 2", sources=[])

        context = memory.get_conversation_context(context_window=2)

        assert "Query 1" in context
        assert "Query 2" in context

    def test_get_previous_entities(self) -> None:
        """Test get_previous_entities retrieves entities from last interaction metadata"""
        memory = ConversationMemory()

        memory.add_interaction(
            query="Tell me about Weyerhaeuser",
            answer="Weyerhaeuser is a timber company",
            sources=[],
            metadata={"entities": ["Weyerhaeuser"]},
        )

        entities = memory.get_previous_entities()

        assert "Weyerhaeuser" in entities

    def test_session_cost_accumulates(self) -> None:
        """Test session cost and token counts accumulate across interactions"""
        memory = ConversationMemory()

        memory.add_interaction(
            query="Q1",
            answer="A1",
            sources=[],
            metadata={"cost_usd": 0.001, "total_tokens": 500},
        )
        memory.add_interaction(
            query="Q2",
            answer="A2",
            sources=[],
            metadata={"cost_usd": 0.002, "total_tokens": 800},
        )

        assert abs(memory.session_cost_usd - 0.003) < 1e-9
        assert memory.session_total_tokens == 1300

    def test_clear_memory(self) -> None:
        """Test clear() resets history and cache"""
        memory = ConversationMemory()

        memory.add_interaction(query="Test", answer="Test", sources=[])
        memory.clear()

        assert len(memory.conversation_history) == 0
        assert len(memory.query_cache) == 0
