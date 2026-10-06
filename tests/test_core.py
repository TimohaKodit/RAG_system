import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def core_module(monkeypatch):
    """Use real local LangChain runnables with no providers or persistent store."""
    from langchain_core.documents import BaseDocumentCompressor, Document
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    state = SimpleNamespace(
        documents=[Document(page_content="Python has lists.", metadata={"user_id": 101})],
        searches=[], search_kwargs=[], reranks=[], rewrites=[], generations=[],
        jev_calls=[], sufficient=True, jev_error=None, leaked_documents=[],
        rewritten_question="How many elements can a Python list have?",
    )

    def model_call(prompt):
        messages = prompt.to_messages()
        if "НЕ ОТВЕЧАЙ" in messages[0].content:
            state.rewrites.append(messages)
            return AIMessage(content=state.rewritten_question)
        state.generations.append(messages)
        return AIMessage(content="Answer from checked context")

    model_factory = Mock(return_value=RunnableLambda(model_call))

    def as_retriever(*, search_kwargs):
        state.search_kwargs.append(search_kwargs)

        def retrieve(query):
            state.searches.append(query)
            # Deliberately return all users to exercise the defensive filter.
            return state.documents

        return RunnableLambda(retrieve)

    class StubCompressor(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            state.reranks.append((query, list(documents)))
            return list(documents) + state.leaked_documents

    compressor_factory = Mock(return_value=StubCompressor())

    class JevError(RuntimeError):
        pass

    def check_context(question, passages):
        state.jev_calls.append((question, list(passages)))
        if state.jev_error is not None:
            raise state.jev_error
        return state.sufficient

    dependency_exports = {
        "dotenv": {"load_dotenv": Mock()},
        "langchain_google_genai": {"ChatGoogleGenerativeAI": model_factory},
        "langchain_cohere": {"CohereRerank": compressor_factory},
        "ingest": {"vector_store": SimpleNamespace(as_retriever=as_retriever)},
        "jev": {"check_context": check_context, "JevError": JevError},
    }
    for name, exports in dependency_exports.items():
        dependency = ModuleType(name)
        for attribute, value in exports.items():
            setattr(dependency, attribute, value)
        monkeypatch.setitem(sys.modules, name, dependency)

    source = Path(__file__).resolve().parents[1] / "core.py"
    spec = importlib.util.spec_from_file_location("_test_rag_core", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module, state=state, Document=Document, JevError=JevError,
        model_factory=model_factory, compressor_factory=compressor_factory,
    )


@pytest.mark.parametrize("sufficient", [True, None])
def test_approved_or_disabled_jev_generates_with_checked_context(core_module, sufficient):
    state = core_module.state
    state.sufficient = sufficient
    result = core_module.module.get_rag_chain(101).invoke({"input": "What is a list?"})

    assert result["answer"] == "Answer from checked context"
    assert result["context"] == state.documents
    assert result["chat_history"] == []
    assert state.searches == ["What is a list?"]
    assert state.jev_calls == [("What is a list?", ["Python has lists."])]
    assert len(state.generations) == 1
    assert "Python has lists." in state.generations[0][0].content
    assert state.rewrites == []
    assert state.reranks == [("What is a list?", state.documents)]
    assert state.search_kwargs == [{"filter": {"user_id": 101}, "k": 10}]
    assert core_module.model_factory.call_args.kwargs["model"] == "gemini-2.5-flash"
    assert core_module.compressor_factory.call_args.kwargs["model"] == "rerank-multilingual-v3.0"


def test_insufficient_context_prevents_generation(core_module):
    state = core_module.state
    state.sufficient = False
    result = core_module.module.get_rag_chain(101).invoke({"input": "Question"})

    assert result["answer"] == core_module.module.INSUFFICIENT_CONTEXT_ANSWER
    assert result["context"] == state.documents
    assert len(state.jev_calls) == 1
    assert state.generations == []


def test_jev_error_propagates_for_safe_api_failure_without_generation(core_module):
    state = core_module.state
    state.jev_error = core_module.JevError("private document and key")
    with pytest.raises(core_module.JevError):
        core_module.module.get_rag_chain(101).invoke({"input": "Question"})

    assert state.generations == []


@pytest.mark.parametrize("contents", [[], ["", "  \n "]])
def test_empty_context_skips_jev_rerank_and_generation(core_module, contents):
    state = core_module.state
    state.documents = [
        core_module.Document(page_content=text, metadata={"user_id": 101})
        for text in contents
    ]
    result = core_module.module.get_rag_chain(101).invoke({"input": "Question"})

    assert result["answer"] == core_module.module.NO_CONTEXT_ANSWER
    assert result["context"] == []
    assert state.jev_calls == []
    assert state.reranks == []
    assert state.generations == []


def test_follow_up_uses_one_rewrite_for_retrieval_and_jev(core_module):
    state = core_module.state
    history = [
        {"role": "user", "content": "Tell me about Python lists"},
        {"role": "ai", "content": "Lists contain elements."},
    ]
    result = core_module.module.get_rag_chain(101).invoke({
        "input": "And how many?", "chat_history": history,
    })

    assert len(state.rewrites) == 1
    assert state.rewrites[0][-1].content == "And how many?"
    assert state.searches == [state.rewritten_question]
    assert state.jev_calls == [(state.rewritten_question, ["Python has lists."])]
    assert state.reranks[0][0] == state.rewritten_question
    assert result["input"] == "And how many?"
    assert result["chat_history"] == history
    assert len(state.generations) == 1
    assert state.generations[0][-1].content == "And how many?"
    assert [message.content for message in state.generations[0][1:-1]] == [
        message["content"] for message in history
    ]


def test_two_users_only_send_their_own_documents_to_providers(core_module):
    state = core_module.state
    first = core_module.Document(page_content="First user text", metadata={"user_id": 101})
    second = core_module.Document(page_content="Second user text", metadata={"user_id": 202})
    state.documents = [first, second]

    first_result = core_module.module.get_rag_chain(101).invoke({"input": "Question"})
    second_result = core_module.module.get_rag_chain(202).invoke({"input": "Question"})

    assert first_result["context"] == [first]
    assert second_result["context"] == [second]
    assert state.jev_calls == [
        ("Question", ["First user text"]), ("Question", ["Second user text"]),
    ]
    assert [documents for _, documents in state.reranks] == [[first], [second]]
    assert "Second user text" not in state.generations[0][0].content
    assert "First user text" not in state.generations[1][0].content
    assert state.search_kwargs == [
        {"filter": {"user_id": 101}, "k": 10},
        {"filter": {"user_id": 202}, "k": 10},
    ]


@pytest.mark.parametrize("metadata", [{}, {"user_id": True}, {"user_id": "1"}])
def test_missing_or_noninteger_user_metadata_is_rejected(core_module, metadata):
    state = core_module.state
    state.documents = [core_module.Document(page_content="Private text", metadata=metadata)]
    result = core_module.module.get_rag_chain(1).invoke({"input": "Question"})

    assert result["context"] == []
    assert state.jev_calls == []
    assert state.reranks == []
    assert state.generations == []


def test_compressor_cannot_inject_other_users_context(core_module):
    state = core_module.state
    state.leaked_documents = [
        core_module.Document(page_content="Other user secret", metadata={"user_id": 202}),
    ]
    result = core_module.module.get_rag_chain(101).invoke({"input": "Question"})

    assert result["context"] == state.documents
    assert state.jev_calls == [("Question", ["Python has lists."])]
    assert "Other user secret" not in state.generations[0][0].content
