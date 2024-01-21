from pathlib import Path

from langchain.prompts import PromptTemplate


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

CONVERSATION_SUMMARY_MEMORY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template=read_prompt_template("./conversation_summary_memory.txt"),
)
