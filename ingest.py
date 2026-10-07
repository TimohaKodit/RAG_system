import os
import unicodedata
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfReadError

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
vector_store = Chroma(embedding_function=embeddings, persist_directory="./db")


class InvalidPDFError(ValueError):
    """PDF повреждён или не содержит доступного для поиска текста."""


def _safe_document_name(value: str | None, fallback: str = "Документ.pdf") -> str:
    """Keep a display filename without paths, controls or URL query secrets."""
    if not isinstance(value, str):
        return fallback
    name = value.split("?", 1)[0].split("#", 1)[0]
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(
        char for char in name
        if not unicodedata.category(char).startswith("C")
    )
    name = name.replace(":", "").strip()[:255]
    return name if name and name not in (".", "..") else fallback


def dc(file_path: str, user_id: int, *, document_name: str | None = None) -> int:
    """Индексировать текст PDF и вернуть число добавленных фрагментов."""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except (ValueError, PdfReadError) as error:
        raise InvalidPDFError(
            "Не удалось прочитать PDF. Файл повреждён или защищён паролем."
        ) from error
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200
    )
    chunks = [
        doc for doc in text_splitter.split_documents(docs)
        if doc.page_content.strip()
    ]
    if not chunks:
        raise InvalidPDFError("В PDF нет текста, доступного для поиска.")

    display_name = _safe_document_name(document_name, _safe_document_name(file_path))
    for doc in chunks:
        doc.metadata["user_id"] = int(user_id)
        doc.metadata["document_name"] = display_name

    attempted_ids: list[str] = []
    try:
        for doc in chunks:
            chunk_id = str(uuid4())
            # Учитываем и текущий фрагмент: запись могла пройти до ошибки.
            attempted_ids.append(chunk_id)
            vector_store.add_documents(documents=[doc], ids=[chunk_id])
    except Exception as indexing_error:
        try:
            vector_store.delete(ids=attempted_ids)
        except Exception:
            indexing_error.add_note(
                "Не удалось откатить фрагменты неуспешной загрузки PDF."
            )
        raise

    return len(chunks)
