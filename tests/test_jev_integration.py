"""HTTP endpoint checks with real RAG and Jev, without provider calls."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from urllib.error import URLError

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def integrated_app(monkeypatch):
    from langchain_core.documents import BaseDocumentCompressor, Document
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    state = SimpleNamespace(
        requests=[], searches=[], generations=[], rewrites=[], reranks=[],
        body={"answers": {"context_sufficient": {"type": "noul", "noul": 0.9}}},
        network_error=None,
        documents=[
            Document(page_content="Private Python notes", metadata={"user_id": 101}),
            Document(page_content="Other user's secret", metadata={"user_id": 202}),
        ],
    )

    def model_call(prompt):
        messages = prompt.to_messages()
        if "НЕ ОТВЕЧАЙ" in messages[0].content:
            state.rewrites.append(messages)
            return AIMessage(content="What does the Python document say?")
        state.generations.append(messages)
        return AIMessage(content="Ответ по проверенному PDF")

    def as_retriever(*, search_kwargs):
        def retrieve(query):
            state.searches.append((query, search_kwargs))
            return state.documents
        return RunnableLambda(retrieve)

    class Compressor(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            state.reranks.append(list(documents))
            return documents

    for name, exports in {
        "dotenv": {"load_dotenv": lambda: None},
        "ingest": {"vector_store": SimpleNamespace(as_retriever=as_retriever)},
        "langchain_google_genai": {
            "ChatGoogleGenerativeAI": lambda **kwargs: RunnableLambda(model_call),
        },
        "langchain_cohere": {"CohereRerank": lambda **kwargs: Compressor()},
    }.items():
        stub = ModuleType(name)
        for attribute, value in exports.items():
            setattr(stub, attribute, value)
        monkeypatch.setitem(sys.modules, name, stub)

    def load_module(name, filename):
        path = Path(__file__).resolve().parents[1] / filename
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    client_module = load_module("jev", "jev.py")

    def fake_urlopen(request, *, timeout):
        state.requests.append(json.loads(request.data.decode("utf-8")))
        if state.network_error is not None:
            raise state.network_error
        return io.BytesIO(json.dumps(state.body).encode("utf-8"))

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    for name in ("JEV_MODEL", "JEV_THRESHOLD", "JEV_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.setenv("TYPESAFE_API_KEY", "stub-key")
    load_module("core", "core.py")
    api_module = load_module("_integration_api", "api.py")
    with TestClient(api_module.app) as client:
        yield client, state


def test_followup_checks_same_owned_context_it_generates(integrated_app):
    client, state = integrated_app
    response = client.post("/ask", json={
        "input": "А что там сказано?", "user_id": 101,
        "chat_history": [{"role": "user", "content": "Расскажи про Python"}],
    })
    assert response.status_code == 200
    assert response.json()["answer"] == "Ответ по проверенному PDF"
    assert isinstance(response.json()["sources"], list)
    assert len(state.rewrites) == len(state.searches) == len(state.requests) == 1
    assert state.requests[0]["state"] == {
        "question": state.searches[0][0], "passages": ["Private Python notes"],
    }
    assert len(state.generations) == 1
    assert "Private Python notes" in state.generations[0][0].content
    assert "Other user's secret" not in state.generations[0][0].content
    assert all(doc.metadata["user_id"] == 101 for doc in state.reranks[0])


@pytest.mark.parametrize("failure", ["transport", "invalid_response", "invalid_config"])
def test_jev_failure_returns_safe_502_without_generation(
    integrated_app, monkeypatch, caplog, failure,
):
    client, state = integrated_app
    if failure == "transport":
        state.network_error = URLError("stub-key Private Python notes")
    elif failure == "invalid_response":
        state.body = {"answers": {"context_sufficient": {"type": "noul", "noul": True}}}
    else:
        monkeypatch.setenv("JEV_THRESHOLD", "NaN")
    response = client.post("/ask", json={"input": "Question", "user_id": 101})
    assert response.status_code == 502
    assert response.json() == {"detail": "Не удалось получить ответ. Попробуй ещё раз позже."}
    assert state.generations == []
    assert "stub-key" not in response.text + caplog.text
    assert "Private Python notes" not in response.text + caplog.text


def test_rejection_is_successful_abstention_without_generation(integrated_app):
    client, state = integrated_app
    state.body["answers"]["context_sufficient"]["noul"] = 0.1
    response = client.post("/ask", json={"input": "Question", "user_id": 101})
    assert response.status_code == 200
    assert "недостаточно информации" in response.json()["answer"]
    assert response.json()["sources"] == []
    assert len(state.requests) == 1
    assert state.generations == []


def test_missing_key_preserves_rag_without_external_jev(integrated_app, monkeypatch):
    client, state = integrated_app
    monkeypatch.delenv("TYPESAFE_API_KEY")
    response = client.post("/ask", json={"input": "Question", "user_id": 101})
    assert response.status_code == 200
    assert state.requests == []
    assert len(state.generations) == 1
