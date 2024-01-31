from dataclasses import dataclass
from typing import Any, TypedDict

from git import Repo
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationSummaryMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline, EmbeddingsFilter)
from langchain.schema import BaseRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from codebase_indexer.api.models import Command
from codebase_indexer.constants import (DEFAULT_OLLAMA_INFERENCE_MODEL,
                                        MAX_SOURCES_WINDOW, OLLAMA_BASE_URL)
from codebase_indexer.rag.agents import create_agent, create_tools
from codebase_indexer.rag.chains import create_search_request_removal_chain
from codebase_indexer.rag.prompts import (
    DEFAULT_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
    REVIEW_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
    TEST_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT)
from codebase_indexer.rag.vector_store import (create_index, load_docs,
                                               load_index)
from codebase_indexer.utils import strip_generated_text


def init_vector_store(
    *,
    repo: Repo,
    sub_folder: str,
    vector_db_dir: str,
) -> VectorStore:
    embeddings_factory = lambda: HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "show_progress_bar": True},
    )

    indexing_args = (
        repo,
        vector_db_dir,
        embeddings_factory,
    )
    db, documents_to_preload = load_index(*indexing_args)
    if db is None:
        db = create_index(
            *indexing_args,
            docs=load_docs(repo, sub_folder, documents_to_preload),
            paths_to_exclude=[
                doc.metadata["file_path"] for doc in documents_to_preload or []
            ],
        )

    return db


class OllamaLLMParams(TypedDict):
    callbacks: list[BaseCallbackHandler]
    inference_model: str


def create_llm(*, params: OllamaLLMParams) -> ChatOllama:
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=params["inference_model"] or DEFAULT_OLLAMA_INFERENCE_MODEL,
        callback_manager=CallbackManager(params["callbacks"] or []),
        verbose=True,
    )


def create_retriever(db: VectorStore):
    retriever_kwargs: dict[str, str | dict[str, Any]] = dict(
        search_type="mmr",
        search_kwargs=dict(k=8, fetch_k=1000),
    )
    return db.as_retriever(**retriever_kwargs)


def create_memory(llm: ChatOllama):
    chat_history = ChatMessageHistory()
    return ConversationSummaryMemory(
        llm=llm,
        chat_memory=chat_history,
        memory_key="chat_history",
        return_messages=True,
    )


def create_conversational_retrieval_chain(
    llm: ChatOllama,
    retriever: BaseRetriever,
    memory: BaseChatMemory | None = None,
    system_prompt: PromptTemplate | None = None,
):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        combine_docs_chain_kwargs={
            "prompt": system_prompt or DEFAULT_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT
        },
        verbose=True,
    )


def create_contextual_retriever(
    embeddings: Embeddings, retriever: VectorStoreRetriever
):
    initial_args = dict(**retriever.search_kwargs)
    retriever.search_kwargs.update(dict(k=20))
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )
    retriever.search_kwargs = initial_args
    return compression_retriever


class RAG:
    @staticmethod
    def create():
        return None, create_conversational_retrieval_chain, None

    @staticmethod
    def extract_commands(llm: ChatOllama, question: str) -> list[Command]:
        command_extraction_tool, _ = create_tools(llm)

        extracted_commands = [
            *filter(
                lambda x: x.lower() != "n/a",  # type: ignore
                map(
                    lambda x: x.strip(),
                    (
                        strip_generated_text(
                            create_agent(llm, [command_extraction_tool])
                            .invoke({"input": question})
                            .get("output", {})
                            .get("text", "")
                            .removeprefix("Answer:")
                        )
                        or ""
                    ).split(", "),
                ),
            )
        ]
        if "search" in extracted_commands:
            extracted_commands.remove("search")
            return ["search", *extracted_commands]  # type: ignore
        if len(extracted_commands) == 0:
            return ["general_chat"]
        return extracted_commands  # type: ignore

    @staticmethod
    def extract_sources_to_search_in(
        llm: ChatOllama, question: str, sources: list[str] | None = None
    ) -> list[str]:
        try:
            _, file_path_extraction_tool = create_tools(llm)
            sources = sources or []


            file_paths = strip_generated_text(
                create_agent(llm, [file_path_extraction_tool])
                .invoke({"input": question})
                .get("output", {})
                .get("text", "")
                .removeprefix("Answer:")
            )
            if file_paths.lower() != "n/a":
                sources.extend([*map(lambda x: x.strip(), file_paths.split(","))])

            return sources
        except ValueError:
            return sources or []

    @staticmethod
    def filter_on_sources(
        retriever: VectorStoreRetriever, sources: list[str]
    ) -> VectorStoreRetriever:
        if len(sources) > 0:
            retriever.search_kwargs.update(dict(filter={"source": {"$in": sources}}))
        return retriever

    @staticmethod
    def cycle_sources_buffer(sources: list[str]) -> list[str]:
        if len(sources) <= MAX_SOURCES_WINDOW:
            return sources
        return sources[len(sources) - MAX_SOURCES_WINDOW : MAX_SOURCES_WINDOW]

    @staticmethod
    def search(retriever: ContextualCompressionRetriever, question: str) -> list[str]:
        docs = retriever.get_relevant_documents(question)

        sources = [doc.metadata["source"] for doc in docs]
        return sources

    @staticmethod
    def remove_search_request(llm: ChatOllama, question: str) -> str:
        question = (
            strip_generated_text(
                create_search_request_removal_chain(llm)
                .invoke({"question": question})
                .get("text", question)
            )
            or question
        )
        return question

    @classmethod
    def from_command(cls, command: Command | None):
        if command == "test":
            return (
                TEST_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
                create_conversational_retrieval_chain,
                None,
            )
        if command == "review":
            return (
                REVIEW_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
                create_conversational_retrieval_chain,
                None,
            )
        if command == "search":
            return (
                None,
                create_conversational_retrieval_chain,
                create_contextual_retriever,
            )
        # TODO(j4ndrw): Need to implement the rest of the commands
        return cls.create()


@dataclass()
class CodebaseIndexer:
    repo: Repo
    vector_db_dir: str | None
    ollama_inference_model: str | None
    db: VectorStore
    llm: ChatOllama
    memory: BaseChatMemory
