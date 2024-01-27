import os
from pathlib import Path

from git import Repo
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from codebase_indexer.api.models import Command, Context
from codebase_indexer.cli.argparser import Args
from codebase_indexer.constants import (
    COMMANDS,
    CONTEXTS,
    DEFAULT_OLLAMA_INFERENCE_MODEL,
    DEFAULT_VECTOR_DB_DIR,
)
from codebase_indexer.rag import (
    RAG,
    create_llm,
    create_memory,
    create_retriever,
    init_vector_store,
)


def cli(args: Args):
    repo_path = Path(os.path.join(os.path.curdir, args.repo_path)).resolve()
    sub_folder = args.sub_folder
    vector_db_dir = args.vector_db_dir or os.path.join(
        DEFAULT_VECTOR_DB_DIR, repo_path.as_posix().rsplit("/")[-1]
    )
    if not repo_path:
        raise Exception("A repository must be specified")

    repo = Repo(repo_path)
    branch = repo.active_branch.name
    commit = repo.head.commit.hexsha

    db = init_vector_store(
        repo_path=repo_path,
        sub_folder=sub_folder or "",
        branch=branch,
        commit=commit,
        vector_db_dir=vector_db_dir,
    )

    llm = create_llm(
        params={
            "inference_model": args.ollama_inference_model
            or DEFAULT_OLLAMA_INFERENCE_MODEL,
            "callbacks": [StreamingStdOutCallbackHandler()],
        }
    )
    memory = create_memory(llm)
    while True:
        question = input(">>> ")
        command: Command | None = None
        contexts: dict[Context, list[str]] = {}
        for command_iter in COMMANDS:
            if question.startswith(f"/{command_iter}"):
                command = command_iter
                question = question.replace(f"/{command}", "").strip()
                break

        # Very ugly, but I can't be asked to implement a proper DSL...
        for context in CONTEXTS:
            if question.startswith(f"@{context}"):
                question = question.replace(f"@{context}", "").strip()
                argument, question = question.split(" ", 1)
                question = question.strip()
                contexts.update({context: [*contexts.get(context, []), argument]})

        system_prompt, qa_chain_factory, custom_retriever_factory = RAG.from_command(
            command
        )

        try:
            retriever = create_retriever(db)
            if len(contexts.get("from", [])) > 0:
                print(contexts["from"])
                retriever.search_kwargs.update(
                    dict(filter={"source": {"$in": contexts["from"]}})
                )
            if command == "search":
                if custom_retriever_factory is not None and db.embeddings is not None:
                    retriever.search_kwargs.update(dict(k=20))
                    retriever = custom_retriever_factory(db.embeddings, retriever)
                    docs = retriever.get_relevant_documents(
                        question, callbacks=llm.callbacks
                    )

                    sources = [doc.metadata["source"] for doc in docs]
                    for source in sources:
                        print(f"\t- {source}")

            else:
                qa = qa_chain_factory(llm, retriever, memory, system_prompt)
                qa.invoke(
                    {
                        "question": question,
                        "chat_history": memory.chat_memory,
                    },
                )
        except KeyboardInterrupt:
            continue
        finally:
            print("\n")
