from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

from codebase_indexer.rag.prompts import QUERY_EXPANSION_PROMPT, SEARCH_REQUEST_REMOVAL


def create_query_expansion_chain(llm: ChatOllama):
    return LLMChain(llm=llm, prompt=QUERY_EXPANSION_PROMPT)


def create_search_request_removal_chain(llm: ChatOllama):
    return LLMChain(llm=llm, prompt=SEARCH_REQUEST_REMOVAL)
