import os
from pathlib import Path
from queue import Queue

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from git import Repo

from codebase_indexer.api import stream
from codebase_indexer.api.cache import codebase_indexers
from codebase_indexer.api.models import Command, Meta
from codebase_indexer.cli import cli
from codebase_indexer.cli.argparser import parse_args
from codebase_indexer.constants import (
    DEFAULT_OLLAMA_INFERENCE_MODEL,
    DEFAULT_VECTOR_DB_DIR,
)
from codebase_indexer.rag import (
    CodebaseIndexer,
    OllamaLLMParams,
    create_llm,
    create_memory,
    init_vector_store,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/register")
async def register(body: Meta):
    repo_path = Path(body.repo_path).resolve()
    sub_folder = body.sub_folder
    vector_db_dir = body.vector_db_dir or os.path.join(repo_path, DEFAULT_VECTOR_DB_DIR)
    if not repo_path:
        raise Exception("A repository must be specified")

    repo = Repo(repo_path)
    branch = repo.active_branch.name
    commit = repo.head.commit.hexsha
    key = repo_path.as_posix()

    db = init_vector_store(
        repo_path=repo_path,
        sub_folder=sub_folder or "",
        branch=branch,
        commit=commit,
        vector_db_dir=vector_db_dir,
    )
    llm = create_llm(
        params=OllamaLLMParams(
            inference_model=body.ollama_inference_model
            or DEFAULT_OLLAMA_INFERENCE_MODEL,
        )
    )
    memory = create_memory(llm)

    codebase_indexers[key] = CodebaseIndexer(
        repo_path=repo_path.as_posix(),
        commit=commit,
        vector_db_dir=vector_db_dir,
        ollama_inference_model=body.ollama_inference_model,
        memory=memory,
        db=db,
        llm=llm,
    )
    return Response(None, 200)


@app.get("/api/ask")
async def ask(
    repo_path: str,
    question: str,
    command: Command | None = None,
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

    stream_generator = stream(
        queue=Queue(),
        command=command,
        repo_path=parsed_repo_path.as_posix(),
        question=question,
    )

    return StreamingResponse(stream_generator, media_type="text/event-stream")


if __name__ == "__main__":
    args = parse_args()
    if args is not None:
        cli(args)
    else:
        uvicorn.run(app, port=11435)
