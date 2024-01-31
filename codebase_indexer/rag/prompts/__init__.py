from pathlib import Path

from langchain.prompts import PromptTemplate

from codebase_indexer.constants import COMMANDS


def read_prompt_template(file_path: str) -> str:
    template = ""
    with open(Path(__file__).parent / file_path) as f:
        template = f.read()
    return template


DEFAULT_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT = PromptTemplate.from_template(
    template=read_prompt_template("./default_conversational_retrieval_chain.txt"),
)

TEST_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT = PromptTemplate.from_template(
    template=read_prompt_template("./test_conversational_retrieval_chain.txt"),
)

REVIEW_CONVERSATIONAL_RETRIEVAL_CHAIN_PROMPT = PromptTemplate.from_template(
    template=read_prompt_template("./review_conversational_retrieval_chain.txt"),
)

CONVERSATION_SUMMARY_MEMORY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template=read_prompt_template("./conversation_summary_memory.txt"),
)

QUERY_EXPANSION_PROMPT = PromptTemplate(
    input_variables=["question"], template=read_prompt_template("./query_expansion.txt")
)

SEARCH_REQUEST_REMOVAL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=read_prompt_template("./search_request_removal.txt"),
)

FILE_PATH_EXTRACTOR_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=read_prompt_template("./file_path_extractor.txt"),
)

COMMAND_EXTRACTOR_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=read_prompt_template("./command_extractor.txt")
    # yuck.
    .replace("{commands}", ", ".join(COMMANDS)),
)
