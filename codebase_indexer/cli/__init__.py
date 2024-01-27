import os
from pathlib import Path

from git import Repo
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from codebase_indexer.api.models import Command
from codebase_indexer.cli.argparser import Args
from codebase_indexer.constants import (
    COMMANDS,
    DEFAULT_OLLAMA_INFERENCE_MODEL,
    DEFAULT_VECTOR_DB_DIR,
)
from codebase_indexer.rag import (
    RAG,
    OllamaLLMParams,
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
        params=OllamaLLMParams(
            inference_model=args.ollama_inference_model
            or DEFAULT_OLLAMA_INFERENCE_MODEL,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    )
    memory = create_memory(llm)
    while True:
        question = input(">>> ")
        command: Command | None = None
        for command_iter in COMMANDS:
            if question.startswith(f"/{command_iter}"):
                command = command_iter
                question = question.replace(f"/{command}", "").lstrip()
                break

        system_prompt, qa_chain_factory, custom_retriever_factory = RAG.from_command(
            command
        )
        retriever = create_retriever(db)

        try:
            if command == "search":
                if custom_retriever_factory is not None and db.embeddings is not None:
                    retriever.search_kwargs.update(dict(k=20))
                    retriever = custom_retriever_factory(db.embeddings, retriever)
                    docs = retriever.get_relevant_documents(
                        question, callbacks=llm.callbacks
                    )

                    memory.chat_memory.add_user_message(question)
                    sources = [doc.metadata["source"] for doc in docs]
                    for source in sources:
                        print(f"\t- {source}")
                    memory.chat_memory.add_ai_message(
                        "\n".join([doc.page_content for doc in docs])
                    )

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
