import os
from os import PathLike
from pathlib import Path
from typing import Callable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.git import GitLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from codebase_indexer.constants import LANGUAGE_FILE_EXTS, LANGUAGES


def collection_name(repo_path: str, commit: str) -> str:
    try:
        return repo_path.split("/")[-1] + commit[:7]
    except IndexError:
        return repo_path.replace("/", "-").replace("~", "")[1:] + commit[:7]


def load_docs(repo_path: PathLike[str], commit: str) -> list[Document]:
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

    loader = GitLoader(repo_path=str(repo_path), branch=commit, file_filter=file_filter)
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
    commit: str,
    vector_db_dir: str,
    embeddings_factory: Callable[[], Embeddings],
) -> tuple[PathLike[str], Chroma]:
    docs = load_docs(repo_path, commit)
    db = Chroma.from_documents(
        docs,
        embeddings_factory(),
        persist_directory=Path(os.path.join(vector_db_dir, commit)).as_posix(),
        collection_name=collection_name(str(repo_path), commit),
    )
    print("Indexed documents")
    return repo_path, db


def load_index(
    repo_path: PathLike[str],
    commit: str,
    vector_db_dir: str,
    embeddings_factory: Callable[[], Embeddings],
) -> tuple[PathLike[str], Chroma] | None:
    store = Chroma(
        collection_name(str(repo_path), commit),
        embeddings_factory(),
        persist_directory=Path(os.path.join(vector_db_dir, commit)).as_posix(),
    )
    result = store.get()
    if len(result["documents"]) == 0:
        return None
    return repo_path, store
