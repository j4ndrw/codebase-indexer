import os

from langchain.prompts import PromptTemplate
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

CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT = """Answer the question in your own words as truthfully as possible from the context given to you.
If you do not know the answer to the question, simply respond with "I don't know. Can you ask another question".
If questions are asked where there is no relevant context available, simply respond with "I don't know. Please ask a question relevant to the documents"
Context: {context}


{chat_history}
Human: {question}
Assistant:"""
CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT = PromptTemplate.from_template(
    CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT,
)

CONVERSATION_SUMMARY_MEMORY_PROMPT = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""
CONVERSATION_SUMMARY_MEMORY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template=CONVERSATION_SUMMARY_MEMORY_PROMPT,
)
