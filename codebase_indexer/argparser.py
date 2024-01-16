from argparse import ArgumentParser
from dataclasses import dataclass

from codebase_indexer.constants import (
    DEFAULT_OLLAMA_EMBEDDINGS_MODEL,
    DEFAULT_OLLAMA_INFERENCE_MODEL,
)


def create_argparser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Indexes your codebase and, given a prompt, provide information about your codebase via an LLM",
    )
    parser.add_argument("repo_path", type=str, help="The path to your repo", default="")
    parser.add_argument(
        "--ollama_inference_model",
        type=str,
        help="The LLM you want to use. Defaults to `mistral-openorca`.",
        default=DEFAULT_OLLAMA_INFERENCE_MODEL,
    )
    parser.add_argument(
        "--ollama_embeddings_model",
        type=str,
        help="The embeddings model you wish to use when creating the index in the vector database. Defaults to `tinyllama`.",
        default=DEFAULT_OLLAMA_EMBEDDINGS_MODEL,
    )
    parser.add_argument(
        "--vector_db_dir",
        type=str,
        help="The path to the vector database. Defaults to `{repo_path}/.llm-index/vectorstores/db`",
        default=None,
    )
    return parser


@dataclass
class Args:
    repo_path: str
    ollama_inference_model: str
    ollama_embeddings_model: str
    vector_db_dir: str | None


def parse_args() -> Args:
    args = create_argparser().parse_args()
    repo_path = args.repo_path
    ollama_inference_model = args.ollama_inference_model
    ollama_embeddings_model = args.ollama_embeddings_model
    vector_db_dir = args.vector_db_dir

    return Args(
        repo_path=repo_path,
        ollama_inference_model=ollama_inference_model,
        ollama_embeddings_model=ollama_embeddings_model,
        vector_db_dir=vector_db_dir,
    )
