import os
from pathlib import Path
from pprint import pprint

from langchain_core.callbacks import StreamingStdOutCallbackHandler

from codebase_indexer.argparser import Args
from codebase_indexer.constants import DEFAULT_VECTOR_DB_DIR
from codebase_indexer.rag import init_llm, init_vector_store


def cli(args: Args):
    repo_path = Path(os.path.join(os.path.curdir, args.repo_path)).resolve()
    vector_db_dir = args.vector_db_dir or os.path.join(repo_path, DEFAULT_VECTOR_DB_DIR)
    if not repo_path:
        raise Exception("A repository must be specified")

    retriever = init_vector_store(repo_path, vector_db_dir)
    _, qa, chat_history = init_llm(
        retriever, [StreamingStdOutCallbackHandler()], args.ollama_inference_model
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
