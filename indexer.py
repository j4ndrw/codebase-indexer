import os
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Callable

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders.git import GitLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

OLLAMA_BASE_URL = "http://" + os.environ["OLLAMA_HOST"]

DEFAULT_VECTOR_DB_DIR = ".llm-index/vectorstores/db"
DEFAULT_OLLAMA_INFERENCE_MODEL = "codellama"
DEFAULT_OLLAMA_EMBEDDINGS_MODEL = "tinyllama"


LANGUAGES = [
    Language.CPP,
    Language.GO,
    Language.JAVA,
    Language.KOTLIN,
    Language.JS,
    Language.TS,
    Language.PHP,
    Language.PROTO,
    Language.PYTHON,
    Language.RST,
    Language.RUBY,
    Language.RUST,
    Language.SCALA,
    Language.SWIFT,
    Language.MARKDOWN,
    Language.LATEX,
    Language.HTML,
    Language.SOL,
    Language.CSHARP,
    Language.COBOL,
]


def collection_name(repo_path: str) -> str:
    return repo_path.rsplit("/", 1)[0].split("/", 1)[1].replace("/", "-")


def prompt_format():
    system_prompt = """You are a senior software engineer, you will use the provided context to answer user questions.
    Read the given context before answering questions and think step by step. If you cannot answer a user question based on
    the provided context, inform the user. Do not use any other information for answering user and do not make up answers. Only answer
    based on real information"""

    instruction = """
    Context: {context}
    User: {question}"""

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "<</SYS>>\n\n"

    return B_INST + B_SYS + system_prompt + E_SYS + instruction + E_INST


QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_format(),
)


def load_docs(repo_path: str, *, path_to_package: str | None = None) -> list[Document]:
    def file_filter(file_path: str) -> bool:
        is_meta = file_path.lower().endswith((".lock", ".md"))
        if not path_to_package:
            return not is_meta
        return not is_meta and file_path.startswith(
            os.path.join(repo_path, path_to_package)
        )

    # TODO: Need to pick the branch from git
    loader = GitLoader(repo_path=repo_path, branch="master", file_filter=file_filter)
    documents = loader.load()
    print("Loaded documents")
    return documents


def split_docs(docs: list[Document]) -> list[Document]:
    splitter: RecursiveCharacterTextSplitter | None = None
    for language in LANGUAGES:
        SplitterClass = splitter or RecursiveCharacterTextSplitter
        splitter = SplitterClass.from_language(
            language, chunk_size=2000, chunk_overlap=200
        )

    if not splitter:
        raise Exception("Splitter is undefined")
    docs = splitter.split_documents(docs)
    print("Split documents")
    return docs


def create_index(
    repo_path: str,
    vector_db_dir: str,
    embeddings_factory: Callable[[], OllamaEmbeddings],
    is_monorepo: bool = False,
) -> tuple[str, Chroma]:
    docs: list[Document] = []
    if is_monorepo:
        path_to_package = input(
            "Which package do you want to load? (E.g. ./packages/ui)"
        )
        docs = load_docs(repo_path, path_to_package=path_to_package)
        repo_path = os.path.join(repo_path, path_to_package)
    else:
        docs = load_docs(repo_path)

    docs = split_docs(docs)
    db = Chroma.from_documents(
        docs,
        embeddings_factory(),
        persist_directory=vector_db_dir,
        collection_name=collection_name(repo_path),
    )
    print("Indexed documents")
    return repo_path, db


def load_index(
    repo_path: str,
    vector_db_dir: str,
    embeddings_factory: Callable[[], OllamaEmbeddings],
    is_monorepo: bool = False,
) -> tuple[str, Chroma] | None:
    if is_monorepo:
        path_to_package = input(
            "Which package do you want to load? (E.g. ./packages/ui)"
        )
        repo_path = os.path.join(repo_path, path_to_package)

    store = Chroma(
        collection_name(repo_path),
        embeddings_factory(),
        persist_directory=vector_db_dir,
    )
    result = store.get()
    if len(result["documents"]) == 0:
        return None
    return repo_path, store


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Indexes your codebase and, given a prompt, provide information about your codebase via an LLM",
    )
    parser.add_argument("repo_path", type=str, help="The path to your repo", default="")
    parser.add_argument(
        "--ollama_inference_model",
        type=str,
        help="The LLM you want to use. Defaults to `codellama`.",
        default=DEFAULT_OLLAMA_INFERENCE_MODEL,
    )
    parser.add_argument(
        "--ollama_embeddings_model",
        type=str,
        help="The embeddings model you wish to use when creating the index in the vector database. Defaults to `tinyllama`.",
        default=DEFAULT_OLLAMA_EMBEDDINGS_MODEL,
    )
    parser.add_argument(
        "--vector_db_dir",
        type=str,
        help="The path to the vector database. Defaults to `{repo_path}/.llm-index/vectorstores/db`",
        default=None,
    )
    parser.add_argument(
        "--monorepo",
        action=BooleanOptionalAction,
        help="Whether you want to index a package of a monorepo or not.",
        default=False,
    )

    args = parser.parse_args()
    repo_path = os.path.abspath(os.path.join(os.path.curdir, args.repo_path))
    ollama_inference_model = args.ollama_inference_model
    ollama_embeddings_model = args.ollama_embeddings_model
    vector_db_dir = args.vector_db_dir or repo_path + DEFAULT_VECTOR_DB_DIR
    is_monorepo = args.monorepo

    if not repo_path:
        raise Exception("A repository must be specified")

    embeddings_factory = lambda: OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=ollama_embeddings_model,
        num_gpu=1,
        num_thread=4,
        num_ctx=5000,
        show_progress=True,
    )

    indexing_args = (
        repo_path,
        vector_db_dir,
        embeddings_factory,
        is_monorepo,
    )
    repo_path, db = load_index(*indexing_args) or create_index(*indexing_args)
    retriever = db.as_retriever(search_type="mmr")

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=ollama_inference_model,
        callback_manager=callback_manager,
        verbose=True,
    )

    while True:
        query = input(">>> ")

        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": QA_CHAIN_PROMPT,
                "memory": ConversationBufferMemory(
                    memory_key="history", input_key="question"
                ),
            },
        )

        docs = retriever.get_relevant_documents(query)
        retrieval_qa.run({"query": query})
        print("\n")
