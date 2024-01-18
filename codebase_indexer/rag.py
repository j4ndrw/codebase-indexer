from os import PathLike

from gpt4all import Embed4All
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationSummaryMemory
from langchain.schema import BaseRetriever
from langchain_community.chat_models import ChatOllama

from codebase_indexer.constants import (
    CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
    DEFAULT_OLLAMA_INFERENCE_MODEL,
    OLLAMA_BASE_URL,
)
from codebase_indexer.embedding import CustomGPT4AllEmbeddings
from codebase_indexer.vector_store import create_index, load_index


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


def init_llm(
    retriever: BaseRetriever,
    callbacks: list[BaseCallbackHandler],
    ollama_inference_model: str | None = None,
):
    chat_history = ChatMessageHistory()
    callback_manager = CallbackManager(callbacks)
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=ollama_inference_model or DEFAULT_OLLAMA_INFERENCE_MODEL,
        callback_manager=callback_manager,
    )

    memory = ConversationSummaryMemory(
        llm=llm,
        chat_memory=chat_history,
        memory_key="chat_history",
        return_messages=True,
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        combine_docs_chain_kwargs={"prompt": CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT},
    )

    return llm, qa, memory.chat_memory
