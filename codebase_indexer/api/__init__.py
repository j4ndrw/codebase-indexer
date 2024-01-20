import asyncio
import os
from pathlib import Path
from queue import Queue
from threading import Thread

from fastapi import Response
from fastapi.responses import StreamingResponse
from git import Repo
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)

from codebase_indexer.api.cache import codebase_indexers
from codebase_indexer.api.models import Command, Meta
from codebase_indexer.api.stream import ChainStreamHandler
from codebase_indexer.constants import DEFAULT_VECTOR_DB_DIR
from codebase_indexer.rag import CodebaseIndexer, Context, RAGBuilder, init_vector_store


async def register(body: Meta):
    repo_path = Path(body.repo_path).resolve()
    vector_db_dir = body.vector_db_dir or os.path.join(repo_path, DEFAULT_VECTOR_DB_DIR)
    if not repo_path:
        raise Exception("A repository must be specified")

    repo = Repo(repo_path)
    branch = repo.active_branch.name

    db = init_vector_store(
        repo_path=repo_path, branch=branch, vector_db_dir=vector_db_dir
    )
    rag_builder = RAGBuilder(db, body.ollama_inference_model)

    codebase_indexers[repo_path.as_posix()] = CodebaseIndexer(
        repo_path=repo_path.as_posix(),
        branch=branch,
        vector_db_dir=vector_db_dir,
        ollama_inference_model=body.ollama_inference_model,
        db=db,
        rag_builder=rag_builder,
    )
    return Response(None, 200)


def generate(qa: BaseConversationalRetrievalChain, **kwargs):
    qa.invoke(kwargs)


async def stream(
    *,
    queue: Queue,
    # TODO(j4ndrw): Need to implement commands
    command: Command | None,
    repo_path: str,
    question: str,
    current_file: str | None = None,
    related_files: list[str] = [],
):
    related_files = related_files if len(related_files) > 0 else []

    rag_builder = codebase_indexers[repo_path].rag_builder
    rag = (
        rag_builder.set_callbacks("qa", [ChainStreamHandler(queue)])
        .set_context(Context(current_file=current_file, related_files=related_files))
        .build()
    )
    if rag is None:
        return

    num_llms_ran = 0

    qa, chat_history = rag
    qa_kwargs = {"question": question, "chat_history": chat_history}
    qa_thread = Thread(target=generate, args=(qa,), kwargs=qa_kwargs)
    qa_thread.start()

    while True:
        if num_llms_ran == 2:
            break

        await asyncio.sleep(0.3)

        value = queue.get()
        if value == None:
            num_llms_ran += 1
        else:
            yield f"data: {value}\n\n"

        queue.task_done()


async def ask(
    repo_path: str,
    question: str,
    command: Command | None = None,
    current_file: str | None = None,
    related_files: str | None = None,
):
    parsed_repo_path = Path(repo_path).resolve()
    if parsed_repo_path not in codebase_indexers:
        await register(
            Meta(
                repo_path=parsed_repo_path.as_posix(),
                vector_db_dir=None,
                ollama_inference_model=None,
            )
        )

    related_files_list = []
    if related_files is not None:
        related_files_list = related_files.split(",")

    stream_generator = stream(
        queue=Queue(),
        command=command,
        repo_path=parsed_repo_path.as_posix(),
        question=question,
        current_file=current_file,
        related_files=related_files_list,
    )

    return StreamingResponse(stream_generator, media_type="text/event-stream")
