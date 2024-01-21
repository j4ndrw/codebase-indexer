import os
from pathlib import Path
from typing import Callable

from git import Repo
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from codebase_indexer.api.models import Command
from codebase_indexer.cli.argparser import Args
from codebase_indexer.constants import (
    COMMANDS,
    DEFAULT_OLLAMA_INFERENCE_MODEL,
    DEFAULT_VECTOR_DB_DIR,
)
from codebase_indexer.rag import (
    LLM,
    RAG,
    Context,
    OllamaLLMParams,
    RAGComponentFactory,
    RAGFactory,
    init_vector_store,
)


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

    memory_params = OllamaLLMParams(
        kind="memory",
        inference_model=args.ollama_inference_model or DEFAULT_OLLAMA_INFERENCE_MODEL,
    )
    memory = RAGComponentFactory.create_memory(LLM(params=memory_params).model)
    default_rag_factory: Callable[[Context | None], RAG] = (
        lambda context: RAGFactory(
            db,
            memory,
            [
                OllamaLLMParams(
                    kind="qa",
                    inference_model=args.ollama_inference_model
                    or DEFAULT_OLLAMA_INFERENCE_MODEL,
                ),
            ],
        )
        .set_context(context)
        .set_callbacks("qa", [StreamingStdOutCallbackHandler()])
        .build()
    )

    specialized_rag_factory: Callable[[Context | None, Command], RAG] = (
        lambda context, command: RAGFactory.from_command(
            db,
            memory,
            [
                OllamaLLMParams(
                    kind="qa",
                    inference_model=args.ollama_inference_model
                    or DEFAULT_OLLAMA_INFERENCE_MODEL,
                ),
            ],
            RAGComponentFactory.from_command(command),
        )
        .set_context(context)
        .set_callbacks("qa", [StreamingStdOutCallbackHandler()])
        .build()
    )

    while True:
        question = input(">>> ")
        current_file = input(
            f" --- Would you like to provide a file for a more exact search?\n    > {repo_path}/"
        )
        context = Context(current_file=current_file) if current_file else None
        command: Command | None = None
        for command_iter in COMMANDS:
            if question.startswith(f"/{command_iter}"):
                command = command_iter
                question = question.replace(f"/{command}", "").lstrip()
                break

        try:
            rag = (
                specialized_rag_factory(context, command)
                if command is not None
                else default_rag_factory(context)
            )
            rag.chain.invoke(
                {"question": question, "chat_history": memory.chat_memory},
            )
        except KeyboardInterrupt:
            continue
        finally:
            print("\n")
