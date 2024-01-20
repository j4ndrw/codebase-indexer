import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import codebase_indexer.api
from codebase_indexer.api.models import Command
from codebase_indexer.cli import cli
from codebase_indexer.cli.argparser import parse_args

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/register")
async def register(body: codebase_indexer.api.Meta):
    return await codebase_indexer.api.register(body)


@app.get("/api/ask")
async def ask(
    repo_path: str,
    question: str,
    command: Command | None = None,
    current_file: str | None = None,
    related_files: str | None = None,
):
    return await codebase_indexer.api.ask(
        command=command,
        repo_path=repo_path,
        question=question,
        current_file=current_file,
        related_files=related_files,
    )


if __name__ == "__main__":
    args = parse_args()
    if args is not None:
        cli(args)
    else:
        uvicorn.run(app, port=11435)
