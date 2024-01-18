from dataclasses import dataclass
from typing import Callable

from langchain.schema import BaseRetriever
from pydantic import BaseModel

from codebase_indexer.rag import RAGBuilder


class Meta(BaseModel):
    repo_path: str
    vector_db_dir: str | None = None
    ollama_inference_model: str | None = None


@dataclass()
class CodebaseIndexer:
    repo_path: str
    branch: str
    vector_db_dir: str | None
    ollama_inference_model: str | None
    retriever: BaseRetriever
    create_rag_builder: Callable[[], RAGBuilder]
