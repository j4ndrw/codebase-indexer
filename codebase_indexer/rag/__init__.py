from os import PathLike
from typing import Literal

from gpt4all import Embed4All
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import \
    BaseConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationSummaryMemory
from langchain.schema import BaseChatMessageHistory, BaseRetriever
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel

from codebase_indexer.constants import (CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
                                        DEFAULT_OLLAMA_INFERENCE_MODEL,
                                        OLLAMA_BASE_URL)
from codebase_indexer.rag.embedding import CustomGPT4AllEmbeddings
from codebase_indexer.rag.vector_store import create_index, load_index


def init_vector_store(
    repo_path: PathLike,
    branch: str,
    vector_db_dir: str,
) -> BaseRetriever:
    embeddings_factory = lambda: CustomGPT4AllEmbeddings(client=Embed4All)

    indexing_args = (repo_path, branch, vector_db_dir, embeddings_factory)
    repo_path, db = load_index(*indexing_args) or create_index(*indexing_args)

    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 1000}
    )

    return retriever


def init_llm(ollama_inference_model: str | None = None):
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=ollama_inference_model or DEFAULT_OLLAMA_INFERENCE_MODEL,
    )

    return llm


def init_rag(
    retriever: BaseRetriever, memory_llm: BaseChatModel, qa_llm: BaseChatModel
):
    chat_history = ChatMessageHistory()
    memory = ConversationSummaryMemory(
        llm=memory_llm,
        chat_memory=chat_history,
        memory_key="chat_history",
        return_messages=True,
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=qa_llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        combine_docs_chain_kwargs={"prompt": CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT},
    )
    return qa, chat_history


class RAGBuilder:
    retriever: BaseRetriever
    memory_llm: BaseChatModel
    qa_llm: BaseChatModel

    memory_llm_callbacks: list[BaseCallbackHandler]
    qa_llm_callbacks: list[BaseCallbackHandler]

    qa: BaseConversationalRetrievalChain
    chat_history: BaseChatMessageHistory

    rag: tuple[BaseConversationalRetrievalChain, BaseChatMessageHistory] | None = None

    def __init__(
        self,
        retriever: BaseRetriever,
        ollama_inference_model: str | None = None,
    ):
        self.retriever = retriever
        self.memory_llm = init_llm(ollama_inference_model)
        self.qa_llm = init_llm(ollama_inference_model)

        self.memory_llm_callbacks = []
        self.qa_llm_callbacks = []

    def set_callbacks(
        self,
        llm_type: Literal["memory"] | Literal["qa"],
        callbacks: list[BaseCallbackHandler],
    ):
        if llm_type == "memory":
            self.memory_llm.callback_manager = CallbackManager(callbacks)
            return self

        if llm_type == "qa":
            self.qa_llm.callback_manager = CallbackManager(callbacks)
            return self

        return self

    def build(self):
        if self.rag is not None:
            return self.rag
        self.rag = init_rag(self.retriever, self.memory_llm, self.qa_llm)
        return self.rag
