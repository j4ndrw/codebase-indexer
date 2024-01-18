import os
from pathlib import Path

from git import Repo
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from codebase_indexer.cli.argparser import Args
from codebase_indexer.constants import DEFAULT_VECTOR_DB_DIR
from codebase_indexer.rag import RAGBuilder, init_vector_store


def cli(args: Args):
    repo_path = Path(os.path.join(os.path.curdir, args.repo_path)).resolve()
    vector_db_dir = args.vector_db_dir or os.path.join(repo_path, DEFAULT_VECTOR_DB_DIR)
    if not repo_path:
        raise Exception("A repository must be specified")

    repo = Repo(repo_path)
    branch = repo.active_branch.name

    retriever = init_vector_store(repo_path, branch, vector_db_dir)
    qa, chat_history = (
        RAGBuilder(retriever, args.ollama_inference_model)
        .set_callbacks("memory", [StreamingStdOutCallbackHandler()])
        .set_callbacks("qa", [StreamingStdOutCallbackHandler()])
        .build()
    )

    while True:
        question = input(">>> ")
        try:
            qa.invoke(
                {"question": question, "chat_history": chat_history},
            )
        except KeyboardInterrupt:
            continue
        finally:
            print("\n")
