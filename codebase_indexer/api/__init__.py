import os
import threading
from pathlib import Path

from fastapi import Response
from fastapi.responses import StreamingResponse
from git import Repo

from codebase_indexer.api.cache import codebase_indexers
from codebase_indexer.api.models import CodebaseIndexer, Meta
from codebase_indexer.api.stream import ChainStreamHandler, ThreadedGenerator
from codebase_indexer.constants import DEFAULT_VECTOR_DB_DIR
from codebase_indexer.rag import RAGBuilder, init_vector_store


async def register(body: Meta):
    repo_path = Path(body.repo_path).resolve()
    vector_db_dir = body.vector_db_dir or os.path.join(repo_path, DEFAULT_VECTOR_DB_DIR)
    if not repo_path:
        raise Exception("A repository must be specified")

    repo = Repo(repo_path)
    branch = repo.active_branch.name

    db = init_vector_store(repo_path, branch, vector_db_dir)
    create_rag_builder = lambda: RAGBuilder(db, body.ollama_inference_model)

    codebase_indexers[repo_path.as_posix()] = CodebaseIndexer(
        repo_path=repo_path.as_posix(),
        branch=branch,
        vector_db_dir=vector_db_dir,
        ollama_inference_model=body.ollama_inference_model,
        db=db,
        create_rag_builder=create_rag_builder,
    )
    return Response(None, 200)


def llm_qa_thread(g: ThreadedGenerator, repo_path: str, question: str):
    try:
        rag = (
            codebase_indexers[repo_path]
            .create_rag_builder()
            .set_callbacks("memory", [ChainStreamHandler(g)])
            .set_callbacks("qa", [ChainStreamHandler(g)])
            .build()
        )
        if rag is None:
            return

        qa, chat_history = rag
        qa.invoke({"question": question, "chat_history": chat_history})
    finally:
        g.close()


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
