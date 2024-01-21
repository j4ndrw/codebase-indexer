import asyncio
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable

from fastapi import Response
from fastapi.responses import StreamingResponse
from git import Repo
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)

from codebase_indexer.api.cache import codebase_indexers
from codebase_indexer.api.models import Command, Meta
from codebase_indexer.api.stream import ChainStreamHandler
from codebase_indexer.constants import (
    DEFAULT_OLLAMA_INFERENCE_MODEL,
    DEFAULT_VECTOR_DB_DIR,
)
from codebase_indexer.rag import (
    LLM,
    CodebaseIndexer,
    Context,
    OllamaLLMParams,
    RAGComponentFactory,
    RAGFactory,
    init_vector_store,
)


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
    memory_params = OllamaLLMParams(
        kind="memory",
        inference_model=body.ollama_inference_model or DEFAULT_OLLAMA_INFERENCE_MODEL,
    )
    memory = RAGComponentFactory.create_memory(LLM(params=memory_params).model)

    default_rag_factory: Callable[
        [Context | None], RAGFactory
    ] = lambda context: RAGFactory(
        db,
        memory,
        [
            OllamaLLMParams(
                kind="qa",
                inference_model=body.ollama_inference_model
                or DEFAULT_OLLAMA_INFERENCE_MODEL,
            ),
        ],
    ).set_context(
        context
    )

    specialized_rag_factory: Callable[
        [Context | None, Command], RAGFactory
    ] = lambda context, command: RAGFactory.from_command(
        db,
        memory,
        [
            OllamaLLMParams(
                kind="qa",
                inference_model=body.ollama_inference_model
                or DEFAULT_OLLAMA_INFERENCE_MODEL,
            ),
        ],
        RAGComponentFactory.from_command(command),
    ).set_context(
        context
    )

    codebase_indexers[repo_path.as_posix()] = CodebaseIndexer(
        repo_path=repo_path.as_posix(),
        branch=branch,
        vector_db_dir=vector_db_dir,
        ollama_inference_model=body.ollama_inference_model,
        memory=memory,
        db=db,
        default_rag_factory=default_rag_factory,
        specialized_rag_factory=specialized_rag_factory,
    )
    return Response(None, 200)


def generate(qa: BaseConversationalRetrievalChain, **kwargs):
    qa.invoke(kwargs)


async def stream(
    *,
    queue: Queue,
    command: Command | None,
    repo_path: str,
    question: str,
    current_file: str | None = None,
    related_files: list[str] = [],
):
    related_files = related_files if len(related_files) > 0 else []

    context = Context(current_file, related_files)
    memory = codebase_indexers[repo_path].memory
    default_rag_factory = codebase_indexers[repo_path].default_rag_factory
    specialized_rag_factory = codebase_indexers[repo_path].specialized_rag_factory
    rag_factory = (
        specialized_rag_factory(context, command)
        if command is not None
        else default_rag_factory(context)
    )
    rag = (
        rag_factory.set_callbacks("qa", [ChainStreamHandler(queue)])
        .set_context(Context(current_file=current_file, related_files=related_files))
        .build()
    )
    if rag is None:
        return

    num_llms_ran = 0

    qa_kwargs = {"question": question, "chat_history": memory.chat_memory}
    qa_thread = Thread(target=generate, args=(rag.chain,), kwargs=qa_kwargs)
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
