from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_structured_chat_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

from codebase_indexer.constants import COMMANDS


def create_tools(llm: ChatOllama):
    command_prompt = PromptTemplate(
        input_variables=["question"],
        template=f"""Only answer the question with the most suitable commands, out of: {[*COMMANDS]}.

        If only one command suits the question, the answer should looks like this: "command_1".
        If multiple commands suit the question, the answer should looks like this: "command_1, command_2, command_3".
        If no command suits the question, simply answer with "N/A".

        Question: {'{question}'}
        Answer: """,
    )
    command_chain = LLMChain(llm=llm, prompt=command_prompt)

    file_path_extractor_prompt = PromptTemplate(
        input_variables=["question"],
        template=f"""Only answer with the file paths provided in the question. If you can't find a file path in the question, answer with "N/A". In your answer, delimit the paths by commas.

        A file path is any string that looks like this: path/to/file

        EXAMPLE START
        Question: Can you review my implementation from those files: path/to/file/a.js, path/to/file/b.rs and path/to/file/c.py
        Answer: path/to/file/a.js, path/to/file/b.rs, path/to/file/c.py
        EXAMPLE END

        EXAMPLE START
        Question: Can you review my implementation from the path/to/file/a.js, path/to/file/b.rs and path/to/file/c.py files?
        Answer: path/to/file/a.js, path/to/file/b.rs, path/to/file/c.py
        EXAMPLE END

        EXAMPLE START
        Question: Can you review my implementation from this file: path/to/file/a.js
        Answer: path/to/file/a.js
        EXAMPLE END

        EXAMPLE START
        Question: Can you review my implementation from path/to/file/a.js?
        Answer: path/to/file/a.js
        EXAMPLE END

        EXAMPLE START
        Question: How do I fix this bug?
        Answer: N/A
        EXAMPLE END

        Question: {'{question}'}
        Answer: """,
    )
    file_path_extractor_chain = LLMChain(llm=llm, prompt=file_path_extractor_prompt)

    command_tool = Tool(
        name="Command classification",
        func=command_chain.invoke,
        return_direct=True,
        description=f"Useful to figure out what command out of the following relates best to the question: {COMMANDS}.",
    )
    file_path_extractor_tool = Tool(
        name="File path",
        func=file_path_extractor_chain.invoke,
        return_direct=True,
        description="Useful to extract file paths from a question, if there are any.",
    )
    return [command_tool, file_path_extractor_tool]


def create_agent(llm: ChatOllama, tools: list[Tool]):
    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent,  # type: ignore
        tools=tools,
        max_iterations=3,
        verbose=False,
    )
    return agent_executor
