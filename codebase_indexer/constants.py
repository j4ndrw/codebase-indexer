import os

from langchain.text_splitter import Language

OLLAMA_BASE_URL = "http://" + os.environ["OLLAMA_HOST"]

DEFAULT_VECTOR_DB_DIR = ".llm-index/vectorstores/db"

DEFAULT_OLLAMA_INFERENCE_MODEL = "mistral-openorca"
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
