import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def ingest_module(monkeypatch):
    """Импортировать ingest без API-запросов и открытия локальной базы."""
    store = SimpleNamespace(
        add_documents=Mock(),
        delete=Mock(),
    )
    loader = SimpleNamespace(load=Mock(return_value=[]))
    loader_class = Mock(return_value=loader)
    splitter = SimpleNamespace(split_documents=Mock(side_effect=lambda docs: docs))
    splitter_class = Mock(return_value=splitter)
    chroma_class = Mock(return_value=store)
    embedding_class = Mock(return_value=object())

    class FakePdfReadError(Exception):
        pass

    dependency_exports = {
        "dotenv": {"load_dotenv": Mock()},
        "pypdf": {},
        "pypdf.errors": {"PdfReadError": FakePdfReadError},
        "langchain_chroma": {"Chroma": chroma_class},
        "langchain_community": {},
        "langchain_community.document_loaders": {"PyPDFLoader": loader_class},
        "langchain_google_genai": {
            "GoogleGenerativeAIEmbeddings": embedding_class,
        },
        "langchain_text_splitters": {
            "RecursiveCharacterTextSplitter": splitter_class,
        },
    }
    for name, exports in dependency_exports.items():
        dependency = ModuleType(name)
        for attribute, value in exports.items():
            setattr(dependency, attribute, value)
        monkeypatch.setitem(sys.modules, name, dependency)

    source = Path(__file__).resolve().parents[1] / "ingest.py"
    spec = importlib.util.spec_from_file_location("_test_pdf_ingest", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module,
        store=store,
        loader=loader,
        loader_class=loader_class,
        splitter_class=splitter_class,
        chroma_class=chroma_class,
        embedding_class=embedding_class,
        pdf_read_error=FakePdfReadError,
    )


def document(text, source="manual.pdf", page=0):
    return SimpleNamespace(
        page_content=text,
        metadata={"source": source, "page": page},
    )


def test_adds_only_text_and_preserves_source_metadata(ingest_module):
    docs = [document("Text", page=2), document(" \n "), document("More", page=4)]
    ingest_module.loader.load.return_value = docs

    count = ingest_module.module.dc("manual.pdf", 17)

    assert count == 2
    assert ingest_module.module.vector_store is ingest_module.store
    ingest_module.loader_class.assert_called_once_with("manual.pdf")
    ingest_module.splitter_class.assert_called_once_with(
        chunk_size=800, chunk_overlap=200
    )
    assert docs[0].metadata == {"source": "manual.pdf", "page": 2, "user_id": 17}
    assert docs[2].metadata == {"source": "manual.pdf", "page": 4, "user_id": 17}
    added = ingest_module.store.add_documents.call_args_list
    assert [call.kwargs["documents"][0] for call in added] == [docs[0], docs[2]]
    ids = [call.kwargs["ids"][0] for call in added]
    assert len(set(ids)) == 2
    ingest_module.store.delete.assert_not_called()


@pytest.mark.parametrize("docs", [[], [document(" \n\t ")]])
def test_empty_pdf_raises_without_writing(ingest_module, docs):
    ingest_module.loader.load.return_value = docs

    with pytest.raises(ingest_module.module.InvalidPDFError, match="нет текста"):
        ingest_module.module.dc("empty.pdf", 17)

    ingest_module.store.add_documents.assert_not_called()
    ingest_module.store.delete.assert_not_called()


@pytest.mark.parametrize("error_kind", ["value_error", "pdf_read_error"])
@pytest.mark.parametrize("stage", ["constructor", "load"])
def test_corrupt_pdf_wraps_only_loader_errors(ingest_module, error_kind, stage):
    error_class = (
        ValueError if error_kind == "value_error" else ingest_module.pdf_read_error
    )
    error = error_class("invalid PDF")
    target = (
        ingest_module.loader_class
        if stage == "constructor"
        else ingest_module.loader.load
    )
    target.side_effect = error

    with pytest.raises(
        ingest_module.module.InvalidPDFError, match="Не удалось прочитать PDF"
    ) as caught:
        ingest_module.module.dc("corrupt.pdf", 17)

    assert caught.value.__cause__ is error
    ingest_module.store.add_documents.assert_not_called()
    ingest_module.store.delete.assert_not_called()


def test_indexing_value_error_is_not_invalid_pdf(ingest_module):
    error = ValueError("embedding service rejected the request")
    ingest_module.loader.load.return_value = [document("Text")]
    ingest_module.store.add_documents.side_effect = error

    with pytest.raises(ValueError) as caught:
        ingest_module.module.dc("manual.pdf", 17)

    assert caught.value is error
    assert not isinstance(caught.value, ingest_module.module.InvalidPDFError)
    attempted_ids = ingest_module.store.add_documents.call_args.kwargs["ids"]
    ingest_module.store.delete.assert_called_once_with(ids=attempted_ids)


def test_failed_upload_rolls_back_only_its_own_ids(ingest_module):
    rows = {"existing-id": document("Existing", source="old.pdf")}
    error = RuntimeError("embedding or storage failed")
    calls = 0

    def add_documents(*, documents, ids):
        nonlocal calls
        calls += 1
        rows[ids[0]] = documents[0]
        if calls == 2:
            # Ошибка возможна даже после фактической записи текущего фрагмента.
            raise error
        return ids

    def delete(*, ids):
        for chunk_id in ids:
            rows.pop(chunk_id, None)

    ingest_module.store.add_documents.side_effect = add_documents
    ingest_module.store.delete.side_effect = delete
    ingest_module.loader.load.return_value = [
        document("First"), document("Second"), document("Third")
    ]

    with pytest.raises(RuntimeError) as caught:
        ingest_module.module.dc("manual.pdf", 17)

    assert caught.value is error
    assert set(rows) == {"existing-id"}
    attempted_ids = [
        call.kwargs["ids"][0]
        for call in ingest_module.store.add_documents.call_args_list
    ]
    ingest_module.store.delete.assert_called_once_with(ids=attempted_ids)
    assert len(attempted_ids) == 2


def test_rollback_failure_preserves_indexing_error(ingest_module):
    error = RuntimeError("indexing failed")
    ingest_module.loader.load.return_value = [document("Text")]
    ingest_module.store.add_documents.side_effect = error
    ingest_module.store.delete.side_effect = RuntimeError("rollback failed")

    with pytest.raises(RuntimeError) as caught:
        ingest_module.module.dc("manual.pdf", 17)

    assert caught.value is error
    assert any("откатить" in note for note in error.__notes__)


def test_two_users_with_same_filename_keep_distinct_ids(ingest_module):
    first_doc = document("First user's text")
    second_doc = document("Second user's text")
    ingest_module.loader.load.side_effect = [[first_doc], [second_doc]]

    assert ingest_module.module.dc("manual.pdf", 17) == 1
    assert ingest_module.module.dc("manual.pdf", 42) == 1

    first_call, second_call = ingest_module.store.add_documents.call_args_list
    assert first_call.kwargs["ids"] != second_call.kwargs["ids"]
    assert first_doc.metadata["user_id"] == 17
    assert second_doc.metadata["user_id"] == 42
    assert type(first_doc.metadata["user_id"]) is int
    assert type(second_doc.metadata["user_id"]) is int


def test_keeps_embedding_model_and_storage_settings(ingest_module):
    assert ingest_module.embedding_class.call_args.kwargs["model"] == (
        "gemini-embedding-2"
    )
    ingest_module.chroma_class.assert_called_once_with(
        embedding_function=ingest_module.module.embeddings,
        persist_directory="./db",
    )
