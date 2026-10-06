from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
INDEX_HTML = PROJECT_DIR / "web" / "index.html"
DATA_DIR = PROJECT_DIR / "data"
MAX_PDF_SIZE = 20 * 1024 * 1024
UPLOAD_READ_SIZE = 1024 * 1024


class ChatMessage(BaseModel):
    role: Literal["user", "ai", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Сообщение истории не должно быть пустым")
        return value


class Query(BaseModel):
    input: str = Field(min_length=1)
    user_id: int = Field(gt=0, strict=True)
    chat_history: list[ChatMessage] = Field(default_factory=list)

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Вопрос не должен быть пустым")
        return value


class Answer(BaseModel):
    answer: str = Field(min_length=1, strict=True)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ответ не должен быть пустым")
        return value


app = FastAPI()


@app.post("/ask", response_model=Answer)
def invoke(inv: Query) -> Answer:
    try:
        from core import get_rag_chain

        chain = get_rag_chain(inv.user_id)
        response = chain.invoke({
            "input": inv.input,
            "chat_history": [message.model_dump() for message in inv.chat_history],
        })
        if not isinstance(response, Mapping):
            raise ValueError("Некорректный результат цепочки")
        return Answer(answer=response.get("answer"))
    except Exception:
        # Exceptions from providers can contain credentials and document text.
        logger.error("Не удалось получить корректный ответ от RAG-цепочки")
        raise HTTPException(
            status_code=502,
            detail="Не удалось получить ответ. Попробуй ещё раз позже.",
        ) from None


class UploadResult(BaseModel):
    chunks: int = Field(gt=0, strict=True)


@app.get("/", response_class=FileResponse, include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


def _remove_failed_upload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Не удалось удалить файл неуспешной загрузки PDF")


@app.post("/upload", response_model=UploadResult)
def upload_pdf(
    file: Annotated[UploadFile, File()],
    user_id: Annotated[int, Form(gt=0)],
) -> UploadResult:
    if (
        Path(file.filename or "").suffix.lower() != ".pdf"
        or file.content_type not in ("application/pdf", "application/octet-stream")
    ):
        raise HTTPException(status_code=400, detail="Загрузи файл в формате PDF.")

    path = DATA_DIR / str(user_id) / f"{uuid4().hex}.pdf"
    created = False
    succeeded = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as destination:
            created = True
            size = 0
            while chunk := file.file.read(UPLOAD_READ_SIZE):
                size += len(chunk)
                if size > MAX_PDF_SIZE:
                    raise HTTPException(
                        status_code=413, detail="PDF должен быть не больше 20 МБ.",
                    )
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="PDF-файл пустой.")

        from ingest import InvalidPDFError, dc

        try:
            chunks = dc(str(path), user_id)
        except InvalidPDFError:
            raise HTTPException(
                status_code=400,
                detail="Не удалось прочитать текст PDF. Проверь файл.",
            ) from None
        result = UploadResult(chunks=chunks)
        succeeded = True
        return result
    except HTTPException:
        raise
    except Exception:
        logger.error("Не удалось загрузить и проиндексировать PDF")
        raise HTTPException(
            status_code=502,
            detail="Не удалось обработать PDF. Попробуй ещё раз позже.",
        ) from None
    finally:
        if created and not succeeded:
            _remove_failed_upload(path)
