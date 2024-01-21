from dataclasses import asdict, astuple, dataclass, field
from os import PathLike
from typing import Any, Callable

from gpt4all import Embed4All
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.memory import ChatMessageHistory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory, BaseRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.language_models import BaseChatModel

from codebase_indexer.api.models import Command, LLMKind
from codebase_indexer.constants import DEFAULT_OLLAMA_INFERENCE_MODEL, OLLAMA_BASE_URL
from codebase_indexer.rag.embedding import CustomGPT4AllEmbeddings
from codebase_indexer.rag.prompts import (
    DEFAULT_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
    TEST_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
)
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


RAGComponentFactoryType = Callable[
    [
        Chroma,
        Context | None,
        ConversationSummaryMemory,
        dict[LLMKind, LLM],
    ],
    RAG,
]


class RAGComponentFactory:
    @staticmethod
    def create_default_retriever(db: Chroma, context: Context | None):
        retriever_kwargs: dict[str, str | dict[str, Any]] = dict(
            search_type="mmr", search_kwargs=dict(k=5, fetch_k=1000)
        )

        related_files = []
        if context is not None:
            collection = db.get()
            metadatas = collection.get("metadatas", [])
            for metadata in metadatas:
                source = metadata.get("source", None)
                if source is None:
                    continue
                for related_file_idx, related_file in enumerate(context.related_files):
                    if related_file in source:
                        context.related_files[related_file_idx] = source

            related_files.extend(context.related_files)

            if context is not None:
                retriever_kwargs["search_kwargs"]["filter"] = {  # type: ignore
                    "source": {
                        "$in": list(
                            filter(
                                None,
                                [
                                    context.current_file,
                                    *context.related_files,
                                ],
                            )
                        )
                    }
                }

        return db.as_retriever(**retriever_kwargs)

    @staticmethod
    def create_memory(llm: BaseChatModel):
        chat_history = ChatMessageHistory()
        return ConversationSummaryMemory(
            llm=llm,
            chat_memory=chat_history,
            memory_key="chat_history",
            return_messages=True,
        )

    @staticmethod
    def create_qa_chain(
        llm: BaseChatModel,
        memory: BaseMemory,
        retriever: BaseRetriever,
        prompt: PromptTemplate | None = None,
    ):
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
            combine_docs_chain_kwargs={
                "prompt": prompt or DEFAULT_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT
            },
            verbose=True,
        )

    @staticmethod
    def default_qa_chain_builder(
        db: Chroma,
        context: Context | None,
        memory: ConversationSummaryMemory,
        llms: dict[LLMKind, LLM],
    ):
        retriever = RAGComponentFactory.create_default_retriever(db, context)
        qa = RAGComponentFactory.create_qa_chain(llms["qa"].model, memory, retriever)
        return RAG(chain=qa)

    @staticmethod
    def test_qa_chain_builder(
        db: Chroma,
        context: Context | None,
        memory: ConversationSummaryMemory,
        llms: dict[LLMKind, LLM],
    ):
        retriever = RAGComponentFactory.create_default_retriever(db, context)
        qa = RAGComponentFactory.create_qa_chain(
            llms["qa"].model,
            memory,
            retriever,
            TEST_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
        )
        return RAG(chain=qa)

    @staticmethod
    def from_command(command: Command) -> RAGComponentFactoryType:
        if command == "test":
            return RAGComponentFactory.test_qa_chain_builder
        # TODO(j4ndrw): Need to implement the rest of the commands
        return RAGComponentFactory.default_qa_chain_builder


class RAGFactory:
    db: Chroma
    memory: ConversationSummaryMemory
    context: Context | None
    llms: dict[LLMKind, LLM]

    rag: RAG

    _component_builder: RAGComponentFactoryType | None

    def __init__(
        self,
        db: Chroma,
        memory: ConversationSummaryMemory,
        llm_params_collection: list[OllamaLLMParams],
        builder: RAGComponentFactoryType | None = None,
    ):
        self.db = db
        self.memory = memory
        self.context = None
        self.llms = {
            llm_params.kind: LLM(params=llm_params)
            for llm_params in llm_params_collection
        }
        self._component_builder = builder

    @classmethod
    def from_command(
        cls,
        db: Chroma,
        memory: ConversationSummaryMemory,
        llm_params_collection,
        builder: RAGComponentFactoryType,
    ):
        return cls(db, memory, llm_params_collection, builder)

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

    def set_context(self, context: Context | None):
        if context is not None:
            self.context = context
        return self

    def build(self) -> RAG:
        component_factory = (
            self._component_builder or RAGComponentFactory.default_qa_chain_builder
        )
        return component_factory(self.db, self.context, self.memory, self.llms)


@dataclass()
class CodebaseIndexer:
    repo_path: str
    branch: str
    vector_db_dir: str | None
    ollama_inference_model: str | None
    db: Chroma
    memory: ConversationSummaryMemory
    default_rag_factory: Callable[[Context | None], RAGFactory]
    specialized_rag_factory: Callable[[Context | None, Command], RAGFactory]
