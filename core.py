import os
import unicodedata
from typing import Any

from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from ingest import vector_store
from jev import check_context


load_dotenv()

contextualixe_q_system_prompt = (
    "Возьми историю чата за последнии вопросы"
    "И переформулируй вопрос исходя из истории сообщений"
    "НЕ ОТВЕЧАЙ на вопрос, только перефразируй его, если нужно."
)

system_prompt = (
    "Ты полезный ассистент, который обучает языку Python"
    "Исппользуйфрагменты из контекста,чтобы отвечать пользователю"
    "Отвечай подробно и развернуто, чтобы твой ответ был максимально понятен"
    "Если информации нет в контексте, просто скажи что не знаешь"
    "Контекст:\n{context}"
)

NO_CONTEXT_ANSWER = (
    "В загруженных документах не найден текст для ответа. "
    "Загрузи подходящий PDF или уточни вопрос."
)
INSUFFICIENT_CONTEXT_ANSWER = (
    "В загруженных документах недостаточно информации для ответа. "
    "Уточни вопрос или загрузи документ с нужными сведениями."
)

contextulize = ChatPromptTemplate.from_messages([
    ("system", contextualixe_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)


def _owned_text_documents(documents: list[Document], user_id: int) -> list[Document]:
    """Keep only nonempty documents with this user's integer metadata."""
    return [
        document for document in documents
        if type(document.metadata.get("user_id")) is int
        and document.metadata["user_id"] == user_id
        and document.page_content.strip()
    ]


def _context_sources(documents: list[Document]) -> list[dict[str, Any]]:
    """Describe the supplied context, without claiming sentence-level citations."""
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    for document in documents:
        name = ""
        candidates = (
            document.metadata.get("document_name"),
            document.metadata.get("source"),
        )
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            candidate = candidate.split("?", 1)[0].split("#", 1)[0]
            candidate = candidate.replace("\\", "/").rsplit("/", 1)[-1]
            candidate = "".join(
                char for char in candidate
                if not unicodedata.category(char).startswith("C")
            ).replace(":", "").strip()[:255]
            if candidate and candidate not in (".", ".."):
                name = candidate
                break
        name = name or "Документ.pdf"
        raw_page = document.metadata.get("page")
        page = (
            raw_page + 1
            if type(raw_page) is int and 0 <= raw_page < 2**53 - 1 else None
        )
        key = (name, page)
        if key not in seen:
            seen.add(key)
            sources.append({"document": name, "page": page})
    return sources[:20]


def _answer_checked_context(
    values: dict[str, Any], config: RunnableConfig,
) -> dict[str, Any]:
    documents = values["context"]
    if not documents:
        return {**values, "answer": NO_CONTEXT_ANSWER, "sources": []}
    sufficient = check_context(
        values["retrieval_question"],
        [document.page_content for document in documents],
    )
    if sufficient is False:
        return {**values, "answer": INSUFFICIENT_CONTEXT_ANSWER, "sources": []}
    answer = question_answer_chain.invoke(values, config=config)
    return {**values, "answer": answer, "sources": _context_sources(documents)}


def get_rag_chain(user_id: int) -> Runnable[dict[str, Any], dict[str, Any]]:
    retriever = vector_store.as_retriever(
        search_kwargs={"filter": {"user_id": user_id}, "k": 10},
    )
    owned_documents = RunnableLambda(
        lambda documents: _owned_text_documents(documents, user_id),
    )
    compressor = CohereRerank(
        model="rerank-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        top_n=10,
    )
    compression = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever | owned_documents,
    )

    # Same branch as create_history_aware_retriever, preserving its query for Jev.
    # A follow-up question is reformulated once and retrieved only once.
    retrieval_question = RunnableBranch(
        (lambda values: not values.get("chat_history"),
         RunnableLambda(lambda values: values["input"])),
        contextulize | llm | StrOutputParser(),
    )
    retrieved_context = (
        RunnableLambda(lambda values: values["retrieval_question"])
        | compression
        | owned_documents
    )
    return (
        RunnablePassthrough.assign(
            chat_history=lambda values: values.get("chat_history", []),
        )
        .assign(retrieval_question=retrieval_question)
        .assign(context=retrieved_context)
        | RunnableLambda(_answer_checked_context)
    ).with_config(run_name="checked_retrieval_chain")
