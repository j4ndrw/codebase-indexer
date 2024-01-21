import os
from pathlib import Path

from git import Repo
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from codebase_indexer.cli.argparser import Args
from codebase_indexer.constants import (
    DEFAULT_OLLAMA_INFERENCE_MODEL,
    DEFAULT_VECTOR_DB_DIR,
)
from codebase_indexer.rag import OllamaLLMParams, RAGFactory, init_vector_store


def cli(args: Args):
    repo_path = Path(os.path.join(os.path.curdir, args.repo_path)).resolve()
    vector_db_dir = args.vector_db_dir or os.path.join(repo_path, DEFAULT_VECTOR_DB_DIR)
    if not repo_path:
        raise Exception("A repository must be specified")

    repo = Repo(repo_path)
    branch = repo.active_branch.name

    db = init_vector_store(
        repo_path=repo_path, branch=branch, vector_db_dir=vector_db_dir
    )
    rag = (
        RAGFactory(
            db,
            OllamaLLMParams(
                kind="memory",
                inference_model=args.ollama_inference_model
                or DEFAULT_OLLAMA_INFERENCE_MODEL,
            ),
            OllamaLLMParams(
                kind="qa",
                inference_model=args.ollama_inference_model
                or DEFAULT_OLLAMA_INFERENCE_MODEL,
            ),
        )
        .set_callbacks("memory", [StreamingStdOutCallbackHandler()])
        .set_callbacks("qa", [StreamingStdOutCallbackHandler()])
        .build()
    )

    while True:
        question = input(">>> ")
        try:
            rag.chain.invoke(
                {"question": question, "chat_history": rag.chat_history},
            )
        except KeyboardInterrupt:
            continue
        finally:
            print("\n")
