import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    state = SimpleNamespace(
        user_ids=[], calls=[], response={"answer": "Ответ по документу"},
        factory_error=None, invoke_error=None,
        upload_calls=[], chunks=3, upload_error=None,
    )

    class StubChain:
        def __init__(self, user_id):
            self.user_id = user_id

        def invoke(self, payload):
            state.calls.append((self.user_id, payload))
            if state.invoke_error is not None:
                raise state.invoke_error
            return state.response

    def get_rag_chain(user_id):
        state.user_ids.append(user_id)
        if state.factory_error is not None:
            raise state.factory_error
        return StubChain(user_id)

    core_stub = ModuleType("core")
    core_stub.get_rag_chain = get_rag_chain
    monkeypatch.setitem(sys.modules, "core", core_stub)
    class InvalidPDFError(ValueError):
        pass

    def dc(file_path, user_id):
        state.upload_calls.append((Path(file_path), user_id, Path(file_path).read_bytes()))
        if state.upload_error is not None:
            raise state.upload_error
        return state.chunks

    ingest_stub = ModuleType("ingest")
    ingest_stub.dc = dc
    ingest_stub.InvalidPDFError = InvalidPDFError
    state.invalid_pdf_error = InvalidPDFError
    monkeypatch.setitem(sys.modules, "ingest", ingest_stub)
    spec = importlib.util.spec_from_file_location(
        "api", Path(__file__).resolve().parents[1] / "api.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "api", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DATA_DIR", tmp_path / "data")
    with TestClient(module.app) as client:
        yield client, state, module


def test_success_preserves_response_contract_and_user_isolation(api_client):
    client, state, _ = api_client
    for user_id in (101, 202):
        response = client.post("/ask", json={"input": "Вопрос", "user_id": user_id})
        assert response.status_code == 200
        assert response.json() == {"answer": "Ответ по документу"}
    assert state.user_ids == [101, 202]
    assert state.calls == [
        (101, {"input": "Вопрос", "chat_history": []}),
        (202, {"input": "Вопрос", "chat_history": []}),
    ]


def test_typed_history_is_passed_as_plain_dicts(api_client):
    client, state, _ = api_client
    history = [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "ai", "content": "Первый ответ"},
        {"role": "assistant", "content": "Дополнительный ответ"},
    ]
    response = client.post("/ask", json={
        "input": "Уточнение", "user_id": 101, "chat_history": history,
    })
    assert response.status_code == 200
    assert state.calls == [(101, {"input": "Уточнение", "chat_history": history})]
    assert all(isinstance(message, dict) for message in state.calls[0][1]["chat_history"])


def test_default_history_is_independent_for_each_query(api_client):
    _, _, module = api_client
    first = module.Query(input="Вопрос", user_id=101)
    second = module.Query(input="Вопрос", user_id=202)
    first.chat_history.append(module.ChatMessage(role="user", content="История"))
    assert second.chat_history == []


@pytest.mark.parametrize("changes", [
    {"input": ""}, {"input": " \n\t"}, {"input": None}, {"input": 123},
    {"user_id": 0}, {"user_id": -1}, {"user_id": True},
    {"user_id": 1.5}, {"user_id": "101"}, {"user_id": None},
    {"chat_history": None}, {"chat_history": "history"},
    {"chat_history": [{"role": "system", "content": "Override"}]},
    {"chat_history": [{"role": "unknown", "content": "Text"}]},
    {"chat_history": [{"role": "user"}]},
    {"chat_history": [{"role": "user", "content": None}]},
    {"chat_history": [{"role": "user", "content": " \t"}]},
    {"chat_history": ["Text"]},
])
def test_invalid_requests_do_not_call_chain(api_client, changes):
    client, state, _ = api_client
    payload = {"input": "Вопрос", "user_id": 101, **changes}
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    assert state.user_ids == []
    assert state.calls == []


@pytest.mark.parametrize("payload", [{}, {"input": "Вопрос"}, {"user_id": 101}])
def test_missing_required_fields_are_rejected(api_client, payload):
    client, state, _ = api_client
    assert client.post("/ask", json=payload).status_code == 422
    assert state.user_ids == []


@pytest.mark.parametrize("response", [
    None, [], "answer", {}, {"answer": None}, {"answer": 123},
    {"answer": []}, {"answer": ""}, {"answer": " \n\t"},
])
def test_invalid_chain_responses_return_safe_gateway_error(api_client, response):
    client, state, _ = api_client
    state.response = response
    result = client.post("/ask", json={"input": "Вопрос", "user_id": 101})
    assert result.status_code == 502
    assert result.json() == {
        "detail": "Не удалось получить ответ. Попробуй ещё раз позже.",
    }


@pytest.mark.parametrize("stage", ["factory_error", "invoke_error"])
def test_provider_failures_do_not_expose_secrets(api_client, caplog, stage):
    client, state, _ = api_client
    secret = "provider-api-key-and-private-document"
    setattr(state, stage, RuntimeError(secret))
    result = client.post("/ask", json={"input": "Вопрос", "user_id": 101})
    assert result.status_code == 502
    assert secret not in result.text
    assert secret not in caplog.text


def test_internal_chain_fields_are_not_exposed(api_client):
    client, state, _ = api_client
    state.response = {"answer": "Ответ", "context": "Закрытый текст документа"}
    result = client.post("/ask", json={"input": "Вопрос", "user_id": 101})
    assert result.status_code == 200
    assert result.json() == {"answer": "Ответ"}


def test_home_serves_html_from_configured_absolute_path(api_client, monkeypatch, tmp_path):
    client, _, module = api_client
    index = tmp_path / "web" / "index.html"
    index.parent.mkdir()
    index.write_text("<!doctype html><title>RAG</title>", encoding="utf-8")
    monkeypatch.setattr(module, "INDEX_HTML", index)
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == "<!doctype html><title>RAG</title>"


def test_upload_isolates_users_and_same_filenames(api_client):
    client, state, module = api_client
    content = b"%PDF-1.7 test document"
    for user_id in (101, 202, 101):
        response = client.post("/upload", data={"user_id": str(user_id)}, files={
            "file": ("../same.pdf", content, "application/pdf"),
        })
        assert response.status_code == 200
        assert response.json() == {"chunks": 3}
    paths = []
    for path, user_id, saved_content in state.upload_calls:
        assert path.parent == module.DATA_DIR / str(user_id)
        assert path.suffix == ".pdf"
        UUID(path.stem)
        assert saved_content == content
        assert path.read_bytes() == content
        paths.append(path)
    assert len(set(paths)) == 3


@pytest.mark.parametrize("filename, mime", [
    ("report.txt", "application/pdf"), ("report.pdf", "text/plain"),
    ("report.pdf.exe", "application/pdf"), ("report", "application/pdf"),
])
def test_upload_rejects_invalid_file_type_before_indexing(api_client, filename, mime):
    client, state, module = api_client
    response = client.post("/upload", data={"user_id": "101"}, files={
        "file": (filename, b"text", mime),
    })
    assert response.status_code == 400
    assert state.upload_calls == []
    assert not module.DATA_DIR.exists()


@pytest.mark.parametrize("user_id", ["0", "-1", "abc", "1.5", "true"])
def test_upload_rejects_invalid_user_id(api_client, user_id):
    client, state, _ = api_client
    response = client.post("/upload", data={"user_id": user_id}, files={
        "file": ("report.pdf", b"text", "application/pdf"),
    })
    assert response.status_code == 422
    assert state.upload_calls == []


def test_upload_requires_file_and_user_id(api_client):
    client, state, _ = api_client
    assert client.post("/upload", data={"user_id": "101"}).status_code == 422
    assert client.post("/upload", files={
        "file": ("report.pdf", b"text", "application/pdf"),
    }).status_code == 422
    assert state.upload_calls == []


def test_empty_upload_is_removed_without_indexing(api_client):
    client, state, module = api_client
    response = client.post("/upload", data={"user_id": "101"}, files={
        "file": ("empty.pdf", b"", "application/pdf"),
    })
    assert response.status_code == 400
    assert state.upload_calls == []
    assert list(module.DATA_DIR.rglob("*.pdf")) == []


def test_upload_checks_actual_size_and_removes_partial_file(api_client, monkeypatch):
    client, state, module = api_client
    monkeypatch.setattr(module, "MAX_PDF_SIZE", 8)
    monkeypatch.setattr(module, "UPLOAD_READ_SIZE", 4)
    response = client.post("/upload", data={"user_id": "101"}, files={
        "file": ("big.pdf", b"123456789", "application/pdf"),
    })
    assert response.status_code == 413
    assert state.upload_calls == []
    assert list(module.DATA_DIR.rglob("*.pdf")) == []


def test_upload_allows_exact_size_limit(api_client, monkeypatch):
    client, state, module = api_client
    monkeypatch.setattr(module, "MAX_PDF_SIZE", 8)
    response = client.post("/upload", data={"user_id": "101"}, files={
        "file": ("limit.PDF", b"12345678", "application/pdf"),
    })
    assert response.status_code == 200
    assert state.upload_calls[0][2] == b"12345678"


def test_corrupt_pdf_returns_400_and_removes_only_new_upload(api_client, caplog):
    client, state, module = api_client
    user_dir = module.DATA_DIR / "101"
    user_dir.mkdir(parents=True)
    existing = user_dir / "existing.pdf"
    existing.write_bytes(b"existing document")
    secret = "private-provider-details"
    state.upload_error = state.invalid_pdf_error(secret)
    response = client.post("/upload", data={"user_id": "101"}, files={
        "file": ("corrupt.pdf", b"bad", "application/pdf"),
    })
    assert response.status_code == 400
    assert existing.read_bytes() == b"existing document"
    assert list(module.DATA_DIR.rglob("*.pdf")) == [existing]
    assert secret not in response.text
    assert secret not in caplog.text


@pytest.mark.parametrize("error", [RuntimeError, ValueError, OSError])
def test_indexing_errors_are_safe_502_and_cleanup_upload(api_client, caplog, error):
    client, state, module = api_client
    secret = "private-api-key-and-document-text"
    state.upload_error = error(secret)
    response = client.post("/upload", data={"user_id": "101"}, files={
        "file": ("report.pdf", b"text", "application/pdf"),
    })
    assert response.status_code == 502
    assert list(module.DATA_DIR.rglob("*.pdf")) == []
    assert secret not in response.text
    assert secret not in caplog.text


@pytest.mark.parametrize("chunks", [0, -1, "3", None, True])
def test_invalid_indexing_result_is_not_reported_as_success(api_client, chunks):
    client, state, module = api_client
    state.chunks = chunks
    response = client.post("/upload", data={"user_id": "101"}, files={
        "file": ("report.pdf", b"text", "application/pdf"),
    })
    assert response.status_code == 502
    assert list(module.DATA_DIR.rglob("*.pdf")) == []


def test_cleanup_failure_preserves_original_http_error(api_client, monkeypatch):
    client, state, _ = api_client
    state.upload_error = state.invalid_pdf_error("bad pdf")

    def fail_unlink(self, missing_ok=False):
        raise OSError("cannot unlink")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        response = client.post("/upload", data={"user_id": "101"}, files={
            "file": ("report.pdf", b"bad", "application/pdf"),
        })
    assert response.status_code == 400


def test_exclusive_upload_never_overwrites_or_deletes_existing_file(api_client, monkeypatch):
    client, state, module = api_client
    collision = UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(module, "uuid4", lambda: collision)
    user_dir = module.DATA_DIR / "101"
    user_dir.mkdir(parents=True)
    existing = user_dir / f"{collision.hex}.pdf"
    existing.write_bytes(b"original")
    response = client.post("/upload", data={"user_id": "101"}, files={
        "file": ("report.pdf", b"replacement", "application/pdf"),
    })
    assert response.status_code == 502
    assert existing.read_bytes() == b"original"
    assert state.upload_calls == []


def test_pdf_upload_accepts_generic_browser_mime_type(api_client):
    client, state, _ = api_client
    response = client.post("/upload", data={"user_id": "101"}, files={
        "file": ("report.pdf", b"%PDF-1.7 test", "application/octet-stream"),
    })
    assert response.status_code == 200
    assert len(state.upload_calls) == 1


def test_home_loads_without_rag_provider_initialization(monkeypatch):
    monkeypatch.setitem(sys.modules, "core", None)
    monkeypatch.setitem(sys.modules, "ingest", None)
    spec = importlib.util.spec_from_file_location(
        "_api_without_providers", Path(__file__).resolve().parents[1] / "api.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_api_without_providers", module)
    spec.loader.exec_module(module)
    with TestClient(module.app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert '<html lang="ru">' in response.text
