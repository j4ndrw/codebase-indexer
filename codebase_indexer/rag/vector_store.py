import os
from os import PathLike
from pathlib import Path
from typing import Callable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.git import GitLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from codebase_indexer.constants import LANGUAGE_FILE_EXTS, LANGUAGES


def collection_name(repo_path: str, sub_folder: str, commit: str) -> str:
    path = os.path.join(repo_path, sub_folder)
    try:
        return path.split("/")[-1] + commit[:7]
    except IndexError:
        return path.replace("/", "-").replace("~", "")[1:] + commit[:7]


def load_docs(repo_path: PathLike[str], sub_folder: str, branch: str) -> list[Document]:
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
        is_build: Callable[[str], bool] = (
            lambda dist: file_path.startswith(f"{dist}/") or f"/{dist}/" in file_path
        )
        is_in_sub_folder = (
            True if not sub_folder else os.path.join(repo_path, sub_folder) in file_path
        )
        return (
            not is_meta
            and not is_build("dist")
            and not is_build("build")
            and is_in_sub_folder
        )

    doc_pool: list[Document] = []
    splitters = {
        language.value: RecursiveCharacterTextSplitter.from_language(
            language, chunk_size=4000, chunk_overlap=200
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
    sub_folder: str,
    branch: str,
    commit: str,
    vector_db_dir: str,
    embeddings_factory: Callable[[], Embeddings],
) -> tuple[PathLike[str], VectorStore]:
    docs = load_docs(repo_path, sub_folder, branch)
    db = Chroma.from_documents(
        docs,
        embeddings_factory(),
        persist_directory=Path(os.path.join(vector_db_dir, branch, commit)).as_posix(),
        collection_name=collection_name(str(repo_path), sub_folder, commit),
    )
    print("Indexed documents")
    return repo_path, db


def load_index(
    repo_path: PathLike[str],
    sub_folder: str,
    branch: str,
    commit: str,
    vector_db_dir: str,
    embeddings_factory: Callable[[], Embeddings],
) -> tuple[PathLike[str], VectorStore] | None:
    store = Chroma(
        collection_name(str(repo_path), sub_folder, commit),
        embeddings_factory(),
        persist_directory=Path(os.path.join(vector_db_dir, branch, commit)).as_posix(),
    )
    result = store.get()
    if len(result["documents"]) == 0:
        return None
    return repo_path, store
