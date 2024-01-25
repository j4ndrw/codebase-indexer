from dataclasses import astuple, dataclass, field
from os import PathLike
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationSummaryMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.schema import BaseRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from codebase_indexer.api.models import Command
from codebase_indexer.constants import DEFAULT_OLLAMA_INFERENCE_MODEL, OLLAMA_BASE_URL
from codebase_indexer.rag.prompts import (
    DEFAULT_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
    REVIEW_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
    TEST_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
)
from codebase_indexer.rag.vector_store import create_index, load_index


def init_vector_store(
    *,
    repo_path: PathLike,
    sub_folder: str,
    branch: str,
    commit: str,
    vector_db_dir: str,
) -> VectorStore:
    embeddings_factory = lambda: HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "show_progress_bar": True},
    )

    indexing_args = (
        repo_path,
        sub_folder,
        branch,
        commit,
        vector_db_dir,
        embeddings_factory,
    )
    repo_path, db = load_index(*indexing_args) or create_index(*indexing_args)

    return db


@dataclass
class OllamaLLMParams:
    callbacks: list[BaseCallbackHandler] = field(default_factory=list)
    inference_model: str = DEFAULT_OLLAMA_INFERENCE_MODEL

    def __iter__(self):
        return iter(astuple(self))


def create_llm(*, params: OllamaLLMParams) -> ChatOllama:
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=params.inference_model,
        callback_manager=CallbackManager(params.callbacks),
        verbose=True,
    )


def create_retriever(db: VectorStore):
    retriever_kwargs: dict[str, str | dict[str, Any]] = dict(
        search_type="mmr", search_kwargs=dict(k=5, fetch_k=1000)
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
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )
    return compression_retriever


class RAG:
    @staticmethod
    def create():
        return None, create_conversational_retrieval_chain, None

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
    repo_path: str
    commit: str
    vector_db_dir: str | None
    ollama_inference_model: str | None
    db: VectorStore
    llm: ChatOllama
    memory: BaseChatMemory
