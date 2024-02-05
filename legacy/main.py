import os
from pathlib import Path
from queue import Queue

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from git import Repo
from langchain.callbacks.manager import CallbackManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from codebase_indexer.api import stream
from codebase_indexer.api.cache import codebase_indexers
from codebase_indexer.api.models import Command, CompleteSchema, MetaSchema
from codebase_indexer.cli import cli
from codebase_indexer.cli.argparser import parse_args
from codebase_indexer.constants import (DEFAULT_OLLAMA_INFERENCE_MODEL,
                                        DEFAULT_VECTOR_DB_DIR, OLLAMA_BASE_URL)
from codebase_indexer.rag import (CodebaseIndexer, create_llm, create_memory,
                                  init_vector_store)

complete_llm: ChatOllama | None = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/register")
async def register(body: MetaSchema):
    repo_path = Path(body.repo_path).resolve()
    sub_folder = body.sub_folder
    vector_db_dir = body.vector_db_dir or os.path.join(
        DEFAULT_VECTOR_DB_DIR, repo_path.as_posix().rsplit("/")[-1]
    )
    if not repo_path:
        raise Exception("A repository must be specified")

    repo = Repo(repo_path)
    key = repo_path.as_posix()

    db = init_vector_store(
        repo=repo,
        sub_folder=sub_folder or "",
        vector_db_dir=vector_db_dir,
    )
    llm = create_llm(
        params={
            "callbacks": [],
            "inference_model": body.ollama_inference_model
            or DEFAULT_OLLAMA_INFERENCE_MODEL,
        }
    )
    memory = create_memory(llm)

    codebase_indexers[key] = CodebaseIndexer(
        repo=repo,
        vector_db_dir=vector_db_dir,
        ollama_inference_model=body.ollama_inference_model,
        memory=memory,
        db=db,
        llm=llm,
    )
    return Response(None, 200)


@app.get("/api/ask")
async def ask(
    repo_path: str,
    question: str,
    command: Command | None = None,
):
    parsed_repo_path = Path(repo_path).resolve()
    if parsed_repo_path not in codebase_indexers:
        await register(
            MetaSchema(
                repo_path=parsed_repo_path.as_posix(),
                vector_db_dir=None,
                ollama_inference_model=None,
            )
        )

    stream_generator = stream(
        queue=Queue(),
        command=command,
        repo_path=parsed_repo_path.as_posix(),
        question=question,
    )

    return StreamingResponse(stream_generator, media_type="text/event-stream")

@app.post("/api/complete")
async def complete(body: CompleteSchema):
    global complete_llm

    if complete_llm is None:
        complete_llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model="starcoder:1b",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
            temperature=0.2,
        )

    embeddings_function = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "show_progress_bar": True},
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    related_buffer_docs = text_splitter.create_documents(body.related_buffers)
    db = Chroma.from_documents(
        related_buffer_docs,
        embeddings_function
    )
    retriever = db.as_retriever()
    related_buffer_docs = [doc.page_content for doc in retriever.get_relevant_documents(body.current_buffer_pre + body.current_buffer_suf)]
    query = f"<fim_prefix>{' '.join(related_buffer_docs)} {body.current_buffer_pre} <fim_suffix>{body.current_buffer_suf} <fim_middle>"
    result = await complete_llm.ainvoke(query)
    content = result.content # type: ignore
    return content

if __name__ == "__main__":
    args = parse_args()
    if args is not None:
        cli(args)
    else:
        uvicorn.run(app, port=11435)
