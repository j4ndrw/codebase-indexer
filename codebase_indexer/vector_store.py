from os import PathLike
from typing import Callable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.git import GitLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

from codebase_indexer.constants import LANGUAGE_FILE_EXTS, LANGUAGES


def collection_name(repo_path: str, branch: str) -> str:
    return repo_path.replace("/", "-").replace("~", "")[1:] + branch


def load_docs(repo_path: PathLike[str], branch: str) -> list[Document]:
    def file_filter(file_path: str) -> bool:
        is_meta = file_path.lower().endswith(
            (
                ".lock",
                ".md",
                "lock.yaml",
                "lock.yml",
                "lock.json",
                ".gitkeep",
                ".gitattributes",
                ".gitignore",
                ".git",
                ".xml",
                ".log",
                ".csv",
                ".jmx",
            )
        )
        return not is_meta

    doc_pool: list[Document] = []
    splitters = {
        language.value: RecursiveCharacterTextSplitter.from_language(
            language, chunk_size=2000, chunk_overlap=200
        )
        for language in LANGUAGES
    }

    loader = GitLoader(repo_path=str(repo_path), branch=branch, file_filter=file_filter)
    docs = loader.load()

    for language in LANGUAGES:
        language_exts = LANGUAGE_FILE_EXTS[language]
        docs_per_lang: list[Document] = []
        for doc in docs:
            if doc.metadata["file_type"][1:] in language_exts:
                docs_per_lang.append(doc)
        doc_pool.extend(splitters[language.value].split_documents(docs_per_lang))

    print("Loaded documents")
    return doc_pool


def create_index(
    repo_path: PathLike,
    branch: str,
    vector_db_dir: str,
    embeddings_factory: Callable[[], GPT4AllEmbeddings],
) -> tuple[PathLike[str], Chroma]:
    docs = load_docs(repo_path, branch)
    db = Chroma.from_documents(
        docs,
        embeddings_factory(),
        persist_directory=vector_db_dir,
        collection_name=collection_name(str(repo_path), branch),
    )
    print("Indexed documents")
    return repo_path, db


def load_index(
    repo_path: PathLike[str],
    branch: str,
    vector_db_dir: str,
    embeddings_factory: Callable[[], GPT4AllEmbeddings],
) -> tuple[PathLike[str], Chroma] | None:
    store = Chroma(
        collection_name(str(repo_path), branch),
        embeddings_factory(),
        persist_directory=vector_db_dir,
    )
    result = store.get()
    if len(result["documents"]) == 0:
        return None
    return repo_path, store
