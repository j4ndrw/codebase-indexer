import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from git import Repo
from langchain.chains.conversational_retrieval.base import \
    BaseConversationalRetrievalChain
from langchain.schema import BaseChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel

from codebase_indexer.argparser import parse_args
from codebase_indexer.cli import cli
from codebase_indexer.constants import DEFAULT_VECTOR_DB_DIR
from codebase_indexer.rag import init_llm, init_vector_store

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)


class Meta(BaseModel):
    repo_path: str
    vector_db_dir: str | None = None
    ollama_inference_model: str | None = None


@dataclass
class CodebaseIndexer:
    repo_path: str
    branch: str
    vector_db_dir: str | None
    ollama_inference_model: str | None
    retriever: VectorStoreRetriever
    lazy_llm: Callable[
        [ThreadedGenerator],
        tuple[ChatOllama, BaseConversationalRetrievalChain, BaseChatMessageHistory],
    ]


codebase_indexers: dict[str, CodebaseIndexer] = dict()


@app.post("/api/register")
async def register(body: Meta):
    repo_path = Path(body.repo_path).resolve()
    vector_db_dir = body.vector_db_dir or os.path.join(repo_path, DEFAULT_VECTOR_DB_DIR)
    if not repo_path:
        raise Exception("A repository must be specified")

    repo = Repo(repo_path)
    branch = repo.active_branch.name

    retriever = init_vector_store(repo_path, branch, vector_db_dir)

    def lazy_llm(g: ThreadedGenerator):
        return init_llm(retriever, [ChainStreamHandler(g)], body.ollama_inference_model)

    codebase_indexers[repo_path.as_posix()] = CodebaseIndexer(
        repo_path=repo_path.as_posix(),
        branch=branch,
        vector_db_dir=vector_db_dir,
        ollama_inference_model=body.ollama_inference_model,
        retriever=retriever,
        lazy_llm=lazy_llm,
    )
    return Response(None, 200)


class Question(BaseModel):
    repo_path: str
    question: str


def llm_qa_thread(g: ThreadedGenerator, repo_path: str, question: str):
    try:
        _, qa, chat_history = codebase_indexers[repo_path].lazy_llm(g)
        qa.invoke({"question": question, "chat_history": chat_history})
    finally:
        g.close()


@app.get("/api/ask")
async def ask(repo_path: str, question: str):
    parsed_repo_path = Path(repo_path).resolve()
    if parsed_repo_path not in codebase_indexers:
        await register(
            Meta(
                repo_path=parsed_repo_path.as_posix(),
                vector_db_dir=None,
                ollama_inference_model=None,
            )
        )

    g = ThreadedGenerator()
    threading.Thread(
        target=llm_qa_thread, args=(g, parsed_repo_path.as_posix(), question)
    ).start()

    return StreamingResponse(g, media_type="text/event-stream")


if __name__ == "__main__":
    args = parse_args()
    if args is not None:
        cli(args)
    else:
        uvicorn.run(app, port=11435)
