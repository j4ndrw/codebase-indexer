import asyncio
from queue import Queue
from threading import Thread

from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)

from codebase_indexer.api.cache import codebase_indexers
from codebase_indexer.api.models import Command
from codebase_indexer.api.stream import ChainStreamHandler
from codebase_indexer.rag import RAG, create_retriever


def generate(
    qa: BaseConversationalRetrievalChain,
    **kwargs,
):
    qa.invoke(kwargs)


async def stream(
    *,
    queue: Queue,
    command: Command | None,
    repo_path: str,
    question: str,
):
    indexer = codebase_indexers[repo_path]

    memory = indexer.memory
    llm = indexer.llm
    llm.callbacks.add_handler(ChainStreamHandler(queue))  # type: ignore

    retriever = create_retriever(indexer.db)
    system_prompt, qa_chain_factory, custom_retriever_factory = RAG.from_command(
        command
    )

    if command == "search":
        if custom_retriever_factory is not None and indexer.db.embeddings is not None:
            retriever.search_kwargs.update(dict(k=20))
            retriever = custom_retriever_factory(indexer.db.embeddings, retriever)
            docs = retriever.get_relevant_documents(question, callbacks=llm.callbacks)

            memory.chat_memory.add_user_message(question)
            sources = [doc.metadata["source"] for doc in docs]
            for source in sources:
                yield f"data: {source}\n\n"
                memory.chat_memory.add_ai_message(
                    "\n".join([doc.page_content for doc in docs])
                )
                await asyncio.sleep(0.1)
        return

    qa = qa_chain_factory(llm, retriever, memory, system_prompt)

    num_llms_ran = 0

    qa_kwargs = {"question": question, "chat_history": memory.chat_memory}
    qa_thread = Thread(target=generate, args=(qa,), kwargs=qa_kwargs)
    qa_thread.start()

    while True:
        if num_llms_ran == 2:
            break

        await asyncio.sleep(0.3)

        value = queue.get()
        if value == None:
            num_llms_ran += 1
        else:
            yield f"data: {value}\n\n"

        queue.task_done()
