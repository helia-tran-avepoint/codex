import os
import re
from typing import Annotated, Any, Callable, Optional, Union

from autogen import (
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    register_function,
)
from autogen.agentchat import ChatResult
from autogen.agentchat.contrib.llamaindex_conversable_agent import (
    LLamaIndexConversableAgent,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from agent_service.agent import MetaAgent
from agent_service.custom_parser import CustomOutputParser
from agent_service.tools import (  # get_executor,
    link_visit_tool,
    retrieve_knowledgebase,
    retrieve_sourcecode,
    run_meta_prompting,
    search_tool,
)
from agent_service.utility import (
    azure_openai_chat_llm,
    azure_openai_llm,
    azure_openai_llm_config,
    code_llm,
    code_llm_config,
    llm,
    llm_config,
)
from shared.utils import DEVELOPMENT
from shared.websocket_server import send_data, send_data_sync


class Team:
    DEFAULT_SYSTEM_MESSAGE = "Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet."

    def __init__(
        self,
        team_name,
    ) -> None:
        if not team_name:
            team_name = self.__class__.__name__
        self.name = re.sub(r"group", "", team_name, flags=re.IGNORECASE)

        self.executor = None
        self.assistant_agent = AssistantAgent(
            name=self.name,
            llm_config=llm_config,
        )
        self.user_proxy = UserProxyAgent(
            name=f"{team_name}UserProxy",
            llm_config=llm_config,
            system_message=self.DEFAULT_SYSTEM_MESSAGE,
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            is_termination_msg=lambda x: x.get("content", "")
            .rstrip()
            .endswith("TERMINATE"),
        )

    def invoke(
        self,
        input: Optional[Union[dict, str, Callable]],
        **kwargs: Any,
    ) -> ChatResult:
        response = self.user_proxy.initiate_chat(
            self.assistant_agent, message=input, silent=DEVELOPMENT
        )
        # while response.get("score", 1.0) < 0.8:
        #     improved_response = self.assistant_agent.initiate_chat(
        #         f"Improve based on critique: {response['feedback']}"
        #     )
        #     response = self.user_proxy.initiate_chat(improved_response)

        # return response if response.get("score", 1.0) >= 0.8 else "TERMINATE"

        return response


class CodeAnalysisTeam(Team):
    def __init__(self, team_name: str) -> None:
        super().__init__(team_name)

        max_iterations = 1

        self.user_proxy = UserProxyAgent(
            name=f"{team_name}UserProxy",
            llm_config=azure_openai_llm_config,  # code_llm_config,
            system_message=(
                "Reply TERMINATE in your response if the task has been solved"
                "You assist users in answering specific questions about code or project metadata. "
                "If the response contains method, MUST list out the full method implementation and its location. "
                "If a query lacks sufficient detail, ask for clarification."
                "Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet."
            ),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
            is_termination_msg=lambda x: x.get("content", "")
            .rstrip()
            .endswith("TERMINATE"),
            silent=DEVELOPMENT,
        )

        self.document_assistant_agent = LLamaIndexConversableAgent(
            name=self.assistant_agent.name,
            llama_index_agent=ReActAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(fn=retrieve_knowledgebase),
                ],
                llm=azure_openai_llm,  # code_llm,
                max_iterations=max_iterations,
                verbose=True,
                output_parser=CustomOutputParser(),
            ),
            system_message=(
                "Answer user questions strictly based on the provided Knowledge. "
                "Do not provide answers that are not directly supported by the Knowledge. "
                "Do not add unrelated content or speculation. "
                "Use tools to find relevant information only when the query is specific enough and necessary. "
                "If the task has been solved at full satisfaction, reply with TERMINATE. "
                "Otherwise, reply with CONTINUE or explain why the task is not solved yet."
            ),
            description="This agent answers code and project-related queries, leveraging tools for retrieving structured data.",
            silent=DEVELOPMENT,
        )

        self.source_code_assistant_agent = LLamaIndexConversableAgent(
            name=self.assistant_agent.name,
            llama_index_agent=ReActAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(fn=retrieve_sourcecode),
                ],
                llm=azure_openai_llm,  # code_llm,
                max_iterations=max_iterations,
                verbose=True,
                output_parser=CustomOutputParser(),
            ),
            system_message=(
                "Answer user questions strictly based on the provided Knowledge. "
                "Do not provide answers that are not directly supported by the Knowledge. "
                "Do not add unrelated content or speculation. "
                "If the location in provided knowledge, must show it in answer. "
                "Use tools to find relevant information only when the query is specific enough and necessary. "
                "If the task has been solved at full satisfaction, reply with TERMINATE. "
                "Otherwise, reply with CONTINUE or explain why the task is not solved yet."
            ),
            description="This agent answers code and project-related queries, leveraging tools for retrieving structured data.",
            silent=DEVELOPMENT,
        )

        agents = [
            self.document_assistant_agent,
            self.source_code_assistant_agent,
            self.user_proxy,
        ]
        group_chat = GroupChat(
            agents=agents,
            speaker_selection_method="round_robin",
            messages=[],
            max_round=len(agents),
        )

        self.group_manager = GroupChatManager(
            groupchat=group_chat, llm_config=azure_openai_llm_config, silent=DEVELOPMENT
        )

    def invoke(
        self,
        input: Optional[Union[dict, str, Callable]],
        **kwargs: Any,
    ) -> ChatResult:
        response = self.user_proxy.initiate_chat(
            self.group_manager,
            message=input,
            summary_method="reflection_with_llm",
            summary_args={
                "summary_prompt": "Your role is to merge responses from multiple agents into a single, coherent answer. Ensure that you eliminate redundant or duplicated information while keeping all **detailed and useful** content intact. Do NOT remove important technical details, but reorganize and integrate responses to create a well-structured and informative reply. Make sure the final response is concise but does not lose valuable insights. You must NOT summarize or shorten responses arbitrarily—your job is to integrate multiple perspectives while keeping full context. Always maintain technical accuracy and completeness in your answers."
            },
            silent=DEVELOPMENT,
        )

        return response


class ResearchTeam(Team):
    ReAct_prompt = """
                    Answer the following questions as best you can. You have access to tools provided.

                    Use the following format:

                    Question: the input question you must answer
                    Thought: you should always think about what to do
                    Action: the action to take
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this process can repeat multiple times)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question

                    Begin!
                    Question: {input}
                    """

    def __init__(self, team_name: str) -> None:
        super().__init__(team_name)

        self.user_proxy = UserProxyAgent(
            name=self.user_proxy.name,
            is_termination_msg=lambda x: x.get("content", "")
            and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="TERMINATE",  # "ALWAYS",
            max_consecutive_auto_reply=1,
            # code_execution_config={"executor": self.executor},
            code_execution_config={"use_docker": ["excutor_with_proxy"]},
        )

        self.assistant_agent = AssistantAgent(
            name=self.assistant_agent.name,
            system_message="Only use the tools you have been provided with. Reply TERMINATE when the task is done.",
            llm_config=llm_config,
        )

        # Register the search tool.
        register_function(
            search_tool,
            caller=self.assistant_agent,
            executor=self.user_proxy,
            name="search_tool",
            description="Search the web for the given query",
        )

        register_function(
            link_visit_tool,
            caller=self.assistant_agent,
            executor=self.user_proxy,
            name="link_visit_tool",
            description="Visit a link and extract its content.",
        )

    # Define the ReAct prompt message. Assuming a "question" field is present in the context
    def react_prompt_message(self, sender, recipient, context):
        return self.ReAct_prompt.format(input=context["question"])

    def invoke(
        self,
        input: Optional[Union[dict, str, Callable]],
        **kwargs: Any,
    ) -> ChatResult:
        response = self.user_proxy.initiate_chat(
            self.assistant_agent,
            message=self.react_prompt_message,
            question=input,
            silent=DEVELOPMENT,
        )
        return response


class ExpertTeam(Team):
    def __init__(self, team_name: str) -> None:
        super().__init__(team_name)
        self.user_proxy = UserProxyAgent(
            name=f"{team_name}UserProxy",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
            default_auto_reply="Continue. If you think the task is solved, please reply me only with 'TERMINATE'.",
        )
        self.user_proxy.register_function(
            function_map={"meta_prompting": lambda **args: run_meta_prompting(**args)}
        )

        self.assistant_agent = MetaAgent(
            name=f"{self.assistant_agent.name}-Meta-Expert",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

    def invoke(
        self,
        input: Optional[Union[dict, str, Callable]],
        **kwargs: Any,
    ) -> ChatResult:
        response = self.user_proxy.initiate_chat(
            self.assistant_agent, message=input, silent=DEVELOPMENT
        )
        return response
