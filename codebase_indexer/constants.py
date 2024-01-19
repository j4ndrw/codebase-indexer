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

LANGUAGE_FILE_EXTS = {
    Language.CPP: ["cpp", "cc", "c", "h", "hpp"],
    Language.GO: ["go", "templ"],
    Language.JAVA: ["java"],
    Language.KOTLIN: ["kt"],
    Language.JS: ["js", "jsx", "cjs", "mjs"],
    Language.TS: ["ts", "tsx"],
    Language.PHP: ["php"],
    Language.PROTO: ["proto"],
    Language.PYTHON: ["py"],
    Language.RST: ["rst"],
    Language.RUBY: ["rb"],
    Language.RUST: ["rs"],
    Language.SCALA: ["scala"],
    Language.SWIFT: ["swift"],
    Language.MARKDOWN: ["md"],
    Language.LATEX: ["tex"],
    Language.HTML: ["html", "cshtml"],
    Language.SOL: ["sol"],
    Language.CSHARP: ["cs", "cshtml"],
    Language.COBOL: ["cbl"],
}
