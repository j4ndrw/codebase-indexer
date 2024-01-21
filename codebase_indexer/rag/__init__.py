from dataclasses import asdict, astuple, dataclass, field
from os import PathLike
from typing import Any

from gpt4all import Embed4All
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.memory import ChatMessageHistory, ConversationSummaryMemory
from langchain.schema import BaseChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.language_models import BaseChatModel

from codebase_indexer.api.models import LLMKind
from codebase_indexer.constants import DEFAULT_OLLAMA_INFERENCE_MODEL, OLLAMA_BASE_URL
from codebase_indexer.rag.embedding import CustomGPT4AllEmbeddings
from codebase_indexer.rag.prompts import CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT
from codebase_indexer.rag.vector_store import create_index, load_index


@dataclass
class Context:
    current_file: str | None = None
    related_files: list[str] = field(default_factory=list)


def init_vector_store(
    *,
    repo_path: PathLike,
    branch: str,
    vector_db_dir: str,
) -> Chroma:
    embeddings_factory = lambda: CustomGPT4AllEmbeddings(client=Embed4All)

    indexing_args = (repo_path, branch, vector_db_dir, embeddings_factory)
    repo_path, db = load_index(*indexing_args) or create_index(*indexing_args)

    return db


@dataclass
class OllamaLLMParams:
    kind: LLMKind = "qa"
    callbacks: list[BaseCallbackHandler] = field(default_factory=list)
    inference_model: str = DEFAULT_OLLAMA_INFERENCE_MODEL

    def __iter__(self):
        return iter(astuple(self))


class LLM:
    model: BaseChatModel
    params: OllamaLLMParams

    def __init__(self, *, params: OllamaLLMParams):
        self.model = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=params.inference_model,
            callback_manager=CallbackManager(params.callbacks),
            verbose=True,
        )
        self.params = params


@dataclass
class RAG:
    chain: BaseConversationalRetrievalChain
    chat_history: BaseChatMessageHistory


class RAGFactory:
    db: Chroma
    context: Context | None
    llms: dict[LLMKind, LLM]

    rag: RAG

    def __init__(self, db: Chroma, *llm_params_collection: OllamaLLMParams):
        self.db = db
        self.context = None
        self.llms = {
            llm_params.kind: LLM(params=llm_params)
            for llm_params in llm_params_collection
        }

    def set_callbacks(
        self,
        llm_kind: LLMKind,
        callbacks: list[BaseCallbackHandler],
    ):
        llm = self.llms[llm_kind]
        params_kwargs = asdict(llm.params)
        params_kwargs["callbacks"] = callbacks
        self.llms[llm_kind] = LLM(params=OllamaLLMParams(**params_kwargs))

        return self

    def set_context(self, context: Context):
        self.context = context
        return self

    def build(self):
        retriever_kwargs: dict[str, str | dict[str, Any]] = dict(
            search_type="mmr", search_kwargs=dict(k=5, fetch_k=1000)
        )

        related_files = []
        if self.context is not None:
            collection = self.db.get()
            metadatas = collection.get("metadatas", [])
            for metadata in metadatas:
                source = metadata.get("source", None)
                if source is None:
                    continue
                for related_file_idx, related_file in enumerate(
                    self.context.related_files
                ):
                    if related_file in source:
                        self.context.related_files[related_file_idx] = source

            related_files.extend(self.context.related_files)

            if self.context is not None:
                retriever_kwargs["search_kwargs"]["filter"] = {  # type: ignore
                    "source": {
                        "$in": list(
                            filter(
                                None,
                                [
                                    self.context.current_file,
                                    *self.context.related_files,
                                ],
                            )
                        )
                    }
                }

        retriever = self.db.as_retriever(**retriever_kwargs)
        chat_history = ChatMessageHistory()
        memory = ConversationSummaryMemory(
            llm=self.llms["memory"].model,
            chat_memory=chat_history,
            memory_key="chat_history",
            return_messages=True,
        )
        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llms["qa"].model,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
            combine_docs_chain_kwargs={"prompt": CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT},
            verbose=True,
        )
        return RAG(chain=qa, chat_history=chat_history)


@dataclass()
class CodebaseIndexer:
    repo_path: str
    branch: str
    vector_db_dir: str | None
    ollama_inference_model: str | None
    db: Chroma
    rag_builder: RAGFactory
