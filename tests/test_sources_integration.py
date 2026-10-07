"""Independent HTTP/source checks; all providers and storage are in memory."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def source_app(monkeypatch, tmp_path):
    # Preload real runnable classes before replacing the ingest splitter module.
    importlib.import_module("langchain_classic.chains.combine_documents")
    importlib.import_module(
        "langchain_classic.retrievers.contextual_compression"
    )
    from langchain_core.documents import BaseDocumentCompressor, Document
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    state = SimpleNamespace(
        documents=[], indexed=[], loaded=[], searches=[], checks=[], generations=[],
        rewrites=[], reranks=[], injections=[], probability=0.9,
        generation_error=None,
    )

    def retrieve_factory(*, search_kwargs):
        def retrieve(query):
            state.searches.append((query, search_kwargs))
            return list(state.documents)
        return RunnableLambda(retrieve)

    def add_documents(*, documents, ids):
        state.indexed.extend(documents)
        state.documents.extend(documents)
        return ids

    store = SimpleNamespace(as_retriever=retrieve_factory,
                            add_documents=add_documents, delete=lambda **kwargs: None)

    class Loader:
        def __init__(self, path):
            self.path = path

        def load(self):
            state.loaded.append(self.path)
            return [Document(page_content=Path(self.path).read_text(),
                             metadata={"source": self.path, "page": 0})]

    class Compressor(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            state.reranks.append(list(documents))
            return list(documents) + state.injections

    def model_call(prompt):
        messages = prompt.to_messages()
        if "НЕ ОТВЕЧАЙ" in messages[0].content:
            state.rewrites.append(messages)
            return AIMessage(content="Standalone follow-up question")
        state.generations.append(messages)
        if state.generation_error is not None:
            raise state.generation_error
        return AIMessage(content="Answer from document")

    exports = {
        "dotenv": {"load_dotenv": lambda: None},
        "langchain_chroma": {"Chroma": lambda **kwargs: store},
        "langchain_community.document_loaders": {"PyPDFLoader": Loader},
        "langchain_text_splitters": {
            "RecursiveCharacterTextSplitter": lambda **kwargs: SimpleNamespace(
                split_documents=lambda documents: documents)},
        "langchain_google_genai": {
            "GoogleGenerativeAIEmbeddings": lambda **kwargs: object(),
            "ChatGoogleGenerativeAI": lambda **kwargs: RunnableLambda(model_call)},
        "langchain_cohere": {"CohereRerank": lambda **kwargs: Compressor()},
    }
    for name, attributes in exports.items():
        stub = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(stub, attribute, value)
        monkeypatch.setitem(sys.modules, name, stub)

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[1] / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("ingest", "ingest.py")
    jev = load("jev", "jev.py")

    def transport(request, *, timeout):
        state.checks.append(json.loads(request.data.decode("utf-8"))["state"])
        return io.BytesIO(json.dumps({"answers": {"context_sufficient": {
            "type": "noul", "noul": state.probability}}}).encode())

    monkeypatch.setattr(jev, "urlopen", transport)
    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.setenv("TYPESAFE_API_KEY", "stub-key")
    for name in ("JEV_THRESHOLD", "JEV_MODEL", "JEV_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    load("core", "core.py")
    api = load("_sources_review_api", "api.py")
    monkeypatch.setattr(api, "DATA_DIR", tmp_path / "data")
    with TestClient(api.app) as client:
        yield SimpleNamespace(client=client, state=state, Document=Document, api=api)


def ask(app, user=101, **extra):
    return app.client.post("/ask", json={"input": "Question", "user_id": user, **extra})


def test_upload_preserves_safe_original_name_and_retrieved_page(source_app):
    app = source_app
    result = app.client.post("/upload", data={"user_id": "101"}, files={
        "file": ("../private/Guide.pdf", b"Guide passage", "application/pdf")})
    assert result.status_code == 200
    assert result.json() == {"chunks": 1}
    assert app.state.indexed[0].metadata["document_name"] == "Guide.pdf"
    response = ask(app)
    assert response.json() == {"answer": "Answer from document", "sources": [
        {"document": "Guide.pdf", "page": 1}]}
    assert "private" not in response.text
    assert "Guide passage" not in response.text


def test_same_upload_name_keeps_distinct_files_and_user_sources(source_app):
    app = source_app
    for user, text in [(101, b"First text"), (202, b"Second text")]:
        result = app.client.post("/upload", data={"user_id": str(user)}, files={
            "file": ("same.pdf", text, "application/pdf")})
        assert result.status_code == 200
    assert len(set(app.state.loaded)) == 2
    for user, text in [(101, "First text"), (202, "Second text")]:
        response = ask(app, user=user)
        assert response.json()["sources"] == [{"document": "same.pdf", "page": 1}]
        assert app.state.checks[-1]["passages"] == [text]
        assert text in app.state.generations[-1][0].content
    assert all(doc.metadata["user_id"] == 101 for doc in app.state.reranks[0])
    assert all(doc.metadata["user_id"] == 202 for doc in app.state.reranks[1])


def test_sources_match_checked_owned_context_and_deduplicate_pages(source_app):
    app = source_app
    app.state.documents = [
        app.Document(page_content="First", metadata={"user_id": 101,
                     "document_name": "C:\\private\\Guide.pdf", "page": 0}),
        app.Document(page_content="Second", metadata={"user_id": 101,
                     "source": "/private/Guide.pdf", "page": 0}),
        app.Document(page_content="Third", metadata={"user_id": 101,
                     "source": "/private/Guide.pdf", "page": 4}),
        app.Document(page_content="Other user", metadata={"user_id": 202,
                     "source": "/private/Other.pdf", "page": 0}),
    ]
    app.state.injections = [app.Document(page_content="Injected other user",
                              metadata={"user_id": 202, "source": "Injected.pdf"})]
    response = ask(app, chat_history=[{"role": "user", "content": "Previous question"}])
    assert response.status_code == 200
    assert response.json()["sources"] == [
        {"document": "Guide.pdf", "page": 1}, {"document": "Guide.pdf", "page": 5}]
    assert app.state.checks == [{"question": "Standalone follow-up question",
                                "passages": ["First", "Second", "Third"]}]
    assert len(app.state.searches) == len(app.state.rewrites) == len(app.state.generations) == 1
    assert "Other user" not in app.state.generations[0][0].content
    assert "private" not in response.text


@pytest.mark.parametrize("page", [None, True, -1, "0", 1.5])
def test_old_metadata_and_invalid_pages_are_compatible(source_app, page):
    app = source_app
    app.state.documents = [app.Document(page_content="Legacy text", metadata={
        "user_id": 101, "source": "F:\\data\\101\\legacy.pdf", "page": page})]
    response = ask(app)
    assert response.status_code == 200
    assert response.json()["sources"] == [{"document": "legacy.pdf", "page": None}]
    assert "data" not in response.text


@pytest.mark.parametrize("mode", ["empty", "rejected"])
def test_no_context_and_rejection_have_no_sources(source_app, mode):
    app = source_app
    if mode == "rejected":
        app.state.documents = [app.Document(page_content="Unrelated passage", metadata={
            "user_id": 101, "source": "irrelevant.pdf", "page": 0})]
        app.state.probability = 0.1
    response = ask(app)
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert app.state.generations == []
    assert len(app.state.checks) == (mode == "rejected")


def test_generator_failure_has_no_success_or_source_payload(source_app, caplog):
    app = source_app
    app.state.documents = [app.Document(page_content="Private text", metadata={
        "user_id": 101, "source": "private.pdf", "page": 0})]
    app.state.generation_error = RuntimeError("Private text stub-key")
    response = ask(app)
    assert response.status_code == 502
    assert set(response.json()) == {"detail"}
    assert "Private text" not in response.text + caplog.text
    assert "stub-key" not in response.text + caplog.text


def test_drive_relative_legacy_name_does_not_break_answer(source_app):
    app = source_app
    app.state.documents = [app.Document(page_content="Legacy text", metadata={
        "user_id": 101, "source": "C:manual.pdf", "page": 0})]
    response = ask(app)
    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1
    assert ":" not in response.json()["sources"][0]["document"]


def test_url_query_and_fragment_are_removed_before_basename(source_app):
    app = source_app
    app.state.documents = [app.Document(page_content="Legacy text", metadata={
        "user_id": 101,
        "source": "https://host/private/manual.pdf?token=private-secret/part",
        "page": 0})]
    response = ask(app)
    assert response.status_code == 200
    assert response.json()["sources"] == [
        {"document": "manual.pdf", "page": 1}]
    assert "private-secret" not in response.text
