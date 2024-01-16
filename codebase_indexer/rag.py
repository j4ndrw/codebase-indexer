import os
from os import PathLike

from gpt4all import Embed4All
from langchain import hub
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from codebase_indexer.constants import DEFAULT_OLLAMA_INFERENCE_MODEL, OLLAMA_BASE_URL
from codebase_indexer.vector_store import create_index, load_index


def init_vector_store(
    repo_path: PathLike,
    vector_db_dir: str,
) -> VectorStoreRetriever:
    embeddings_factory = lambda: GPT4AllEmbeddings(client=Embed4All)

    indexing_args = (repo_path, vector_db_dir, embeddings_factory)
    repo_path, db = load_index(*indexing_args) or create_index(*indexing_args)

    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}
    )

    return retriever


def init_llm(
    retriever: VectorStoreRetriever,
    ollama_inference_model: str | None = None,
    *callbacks: BaseCallbackHandler
):
    callback_manager = CallbackManager([*callbacks])
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=ollama_inference_model or DEFAULT_OLLAMA_INFERENCE_MODEL,
        callback_manager=callback_manager,
        num_ctx=8192,
        num_gpu=1,
    )
    memory = ConversationSummaryMemory(
        llm=llm, return_messages=True, memory_key="chat_history"
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )
    return llm, qa
