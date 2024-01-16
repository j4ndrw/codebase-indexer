import os
from pathlib import Path

from codebase_indexer.argparser import parse_args
from codebase_indexer.constants import DEFAULT_VECTOR_DB_DIR
from codebase_indexer.rag import init_llm, init_vector_store


def cli():
    args = parse_args()

    repo_path = Path(os.path.join(os.path.curdir, args.repo_path)).resolve()
    vector_db_dir = args.vector_db_dir or os.path.join(repo_path, DEFAULT_VECTOR_DB_DIR)
    if not repo_path:
        raise Exception("A repository must be specified")

    retriever = init_vector_store(repo_path, vector_db_dir)
    _, qa = init_llm(retriever, args.ollama_inference_model)

    while True:
        query = input(">>> ")
        qa(query)
        print("\n")
