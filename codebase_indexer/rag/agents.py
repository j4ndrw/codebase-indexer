from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

from codebase_indexer.constants import COMMANDS
from codebase_indexer.rag.prompts import (COMMAND_EXTRACTOR_PROMPT,
                                          FILE_PATH_EXTRACTOR_PROMPT)


def create_tools(llm: ChatOllama):
    command_chain = LLMChain(llm=llm, prompt=COMMAND_EXTRACTOR_PROMPT)
    file_path_extractor_chain = LLMChain(llm=llm, prompt=FILE_PATH_EXTRACTOR_PROMPT)

    command_tool = Tool(
        name="Classify command",
        func=command_chain.invoke,
        return_direct=True,
        description=f"Useful to figure out what command out of the following relates best to the question: {COMMANDS}.",
    )
    file_path_extractor_tool = Tool(
        name="Extract file path",
        func=file_path_extractor_chain.invoke,
        return_direct=True,
        description="Useful to extract file paths from a question, if there are any.",
    )
    return [command_tool, file_path_extractor_tool]


def create_agent(llm: ChatOllama, tools: list[Tool]):
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent,  # type: ignore
        tools=tools,
        max_iterations=3,
        verbose=True,
        return_intermediate_steps=True
    )
    return agent_executor
