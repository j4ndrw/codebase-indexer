import os
from pathlib import Path

from git import Repo
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from codebase_indexer.cli.argparser import Args
from codebase_indexer.constants import (
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

    db = init_vector_store(
        sub_folder=sub_folder or "",
        repo=repo,
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
    sources: list[str] = []
    while True:
        question = input(">>> ")

        commands = RAG.extract_commands(llm, question)
        sources = RAG.extract_sources_to_search_in(llm, question, sources)

        for command in commands:
            if command == "new_conversation":
                sources = []
                continue

            system_prompt, qa_chain_factory, custom_retriever_factory = (
                RAG.from_command(command)
            )

            try:
                retriever = create_retriever(db)
                if command == "search":
                    if (
                        custom_retriever_factory is not None
                        and db.embeddings is not None
                    ):
                        embeddings_retriever = custom_retriever_factory(
                            db.embeddings, retriever
                        )
                        sources = RAG.search(embeddings_retriever, question)
                        for source in sources:
                            print(f"\t- {source}")

                        sources.extend(sources)
                        question = RAG.remove_search_request(llm, question)

                else:
                    retriever = RAG.filter_on_sources(retriever, sources)
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

        RAG.cycle_sources_buffer(sources)
