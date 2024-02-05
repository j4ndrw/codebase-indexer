import os
from os import PathLike
from pathlib import Path
from typing import Callable

from git import Commit, Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.git import GitLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from codebase_indexer.constants import LANGUAGE_FILE_EXTS, LANGUAGES


def load_docs(
    repo: Repo, sub_folder: str, preloaded_documents: list[Document] | None = None
) -> list[Document]:
    preloaded_documents = preloaded_documents or []
    paths_to_exclude: list[str] = [
        doc.metadata["file_path"] for doc in preloaded_documents
    ]
    repo_path: PathLike[str] = repo.working_dir  # type: ignore
    branch = repo.active_branch.name

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
            and file_path not in paths_to_exclude
        )

    doc_pool: list[Document] = []
    splitters = {
        language.value: RecursiveCharacterTextSplitter.from_language(
            language, chunk_size=4000, chunk_overlap=200
        )
        for language in LANGUAGES
    }

    loader = GitLoader(repo_path=str(repo_path), branch=branch, file_filter=file_filter)
    docs = [*preloaded_documents, *loader.load()]

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
    repo: Repo,
    vector_db_dir: str,
    embeddings_factory: Callable[[], Embeddings],
    *,
    docs: list[Document],
    paths_to_exclude: list[str] | None = None,
) -> VectorStore:
    paths_to_exclude = paths_to_exclude or []

    docs_to_index = [
        doc for doc in docs if doc.metadata["file_path"] not in paths_to_exclude
    ]
    for doc in docs_to_index:
        doc.metadata.update({"commit": repo.head.commit.hexsha})

    repo_path: PathLike[str] = repo.working_dir  # type: ignore
    repo_name: str = os.path.join(*Path(repo_path).as_posix().rsplit("/", 2)[1:])

    db = Chroma.from_documents(
        docs_to_index,
        embeddings_factory(),
        persist_directory=Path(os.path.join(vector_db_dir, repo_name)).as_posix(),
        collection_name=repo_name.replace("/", "_"),
    )

    if len(docs_to_index) > 0:
        db._collection.delete(
            where={  # type: ignore
                "$and": [
                    {"commit": {"$ne": repo.head.commit.hexsha}},
                    {
                        "file_path": {
                            "$in": [doc.metadata["file_path"] for doc in docs_to_index]
                        }
                    },
                ]
            }
        )
        collection = db.get()
        metadatas = collection.get("metadatas", [])
        for metadata in metadatas:
            metadata.update({"commit": repo.head.commit.hexsha})
        new_docs = [
            Document(page_content=page_content, metadata=metadata)
            for (page_content, metadata) in zip(
                collection.get("documents", []), metadatas
            )
        ]
        for id, doc in zip(collection.get("ids", []), new_docs):
            db.update_document(id, doc)

    print("Indexed documents")
    return db


def load_index(
    repo: Repo,
    vector_db_dir: str,
    embeddings_factory: Callable[[], Embeddings],
) -> tuple[VectorStore | None, list[Document] | None]:
    repo_path: PathLike[str] = repo.working_dir  # type: ignore
    repo_name: str = os.path.join(*Path(repo_path).as_posix().rsplit("/", 2)[1:])
    store = Chroma(
        embedding_function=embeddings_factory(),
        persist_directory=Path(os.path.join(vector_db_dir, repo_name)).as_posix(),
        collection_name=repo_name.replace("/", "_"),
    )
    current_commit = repo.head.commit

    all_result = store.get()
    if len(all_result.get("documents", [])) == 0:
        return None, None

    current_commit_result = store.get(where={"commit": current_commit.hexsha})
    if len(current_commit_result.get("documents", [])) != 0:
        return store, None

    previous_commit: Commit | None = None
    for metadata in all_result["metadatas"]:
        if metadata["commit"] != current_commit.hexsha:
            previous_commit = repo.commit(metadata["commit"])

    if previous_commit is None:
        return None, None

    changed_files: list[str] = []

    for diff in current_commit.diff(previous_commit):
        if diff.a_blob is not None and diff.a_blob.path not in changed_files:
            changed_files.append(diff.a_blob.path)

        if diff.b_blob is not None and diff.b_blob.path not in changed_files:
            changed_files.append(diff.b_blob.path)

    result_to_preload = store.get(where={"source": {"$nin": changed_files}})  # type: ignore

    documents_to_preload = [
        Document(page_content=page_content, metadata=metadata)
        for page_content, metadata in zip(
            result_to_preload["documents"], result_to_preload["metadatas"]
        )
    ]

    return None, documents_to_preload
