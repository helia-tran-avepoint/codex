import asyncio
from os import name
from typing import Literal, Optional, TypedDict

import urllib3
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, ValidationError, field_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from agent_service.team import CodeAnalysisTeam, ExpertTeam, ResearchTeam
from agent_service.utility import (
    azure_openai_chat_llm,
    chat_code_llm,
    chat_general_llm,
    chat_llm,
    chat_routing_llm,
)
from shared.utils import DEVELOPMENT
from shared.websocket_server import send_data, send_data_sync

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CodeAnalysisTeamName = CodeAnalysisTeam.__name__
ResearchTeamName = ResearchTeam.__name__
ExpertTeamName = ExpertTeam.__name__

Nodes = (
    # "General",
    CodeAnalysisTeamName,
    # ResearchTeamName,
    # ExpertTeamName,
    # "HumanInvolved",
)

NodeDescriptions = (
    # "For greetings, casual conversations, or non-specific queries.",
    "For questions related to coding, debugging, code analysis, troubleshooting, customer issue, business logic in leogcc project or software development.",
    # "For questions requiring web searches or external information gathering.",
    # "For highly specialized questions requiring domain-specific expertise.",
    # "For unclear intent should involve human in loop to ask human some more detail information so that can get a clear intent to make to progress go ahead.",
)


class Route(BaseModel):
    """Decide where to go next"""

    goto: Optional[Literal[Nodes]] = None  # type: ignore

    @field_validator("goto")
    def validate_goto(cls, value):
        if value not in Nodes:
            raise ValueError(f"Invalid goto value: {value}")
        return value


class EmptyResponseError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    retry=retry_if_exception_type(EmptyResponseError),
    retry_error_callback=lambda retry_state: ExpertTeamName,
)
def route_node(
    state: MessagesState,
) -> Literal[Nodes]:  # type: ignore
    send_data_sync(
        {
            "result": "[Route Agent]: Given the nature of this request, it aligns best with the 'Code Analysis Team', which specializes in project analysis, debugging, troubleshooting, and code review.\n"
        }
    )
    return "CodeAnalysisTeam"

    last_message = state["messages"][-1].content

    response = azure_openai_chat_llm.with_structured_output(
        Route, method="json_mode"
    ).invoke(
        [
            {
                "role": "system",
                "content": f'''
                    You are a router agent. **Always return a valid JSON response.**

                    - If the user input **clearly belongs to a team**, return `{ {"goto": "CodeAnalysisTeam"} }`
                        -- Here are the available agents/team:
                            {"\n\n".join([f"Agent Name: {k}\nAgent Description: {v}\n" for k, v in zip(Nodes, NodeDescriptions)])}
                    
                    **All responses must be in JSON format!**
                ''',
            },
            {"role": "user", "content": last_message},
        ]
    )
    # - If you can **answer the question directly**, return `{ {"answer": "your answer here"} }`
    # - If the input is **unclear**, return `{ {"clarification": "Can you provide more details?"} }`

    return response.goto  # type: ignore


translator_agent = create_react_agent(
    chat_llm,
    tools=[],
    state_modifier="You are an agent specializing in translating",
)

general_agent = create_react_agent(
    chat_general_llm,
    tools=[],
    state_modifier="""
                        You are a friendly and helpful assistant. Your role is to provide polite and concise responses to general questions or greetings from the user.

                        When responding:
                        1. If the user greets you (e.g., "Hi", "Hello", "Good morning"), respond with a friendly greeting.
                        2. If the user asks a general question, provide a concise and helpful answer. If unsure, suggest where the user can find more information.
                        3. Always keep your tone positive and approachable.

                        Format your response directly as a conversational message. Do not include unnecessary details or technical jargon unless explicitly requested.

                        Examples:
                        - User: "Hello"
                        Response: "Hello! How can I assist you today?"
                        - User: "What's the weather like today?"
                        Response: "I'm not sure about the weather, but you can check it online for your location."

                        Now, generate a response based on the user's input.
                    """,
)

code_analysis_team = CodeAnalysisTeam(CodeAnalysisTeamName)
research_team = ResearchTeam(ResearchTeamName)
expert_team = ExpertTeam(ExpertTeamName)


def code_analysis_group_node(state: MessagesState):
    # full_context = [
    #     {
    #         "role": "user" if msg["role"] == "human" else "assistant",
    #         "content": msg["content"],
    #     }
    #     for msg in state["messages"]
    # ]

    role_map = {
        HumanMessage: "user",
        AIMessage: "assistant",
    }
    full_context = []

    for msg in state["messages"]:
        for _t, name in role_map.items():
            if isinstance(msg, _t):
                full_context.append(
                    {
                        "role": name,
                        "content": msg.content,
                    }
                )

    # last_message = state["messages"][-1]
    response = code_analysis_team.invoke(full_context[-1])  # type: ignore
    # content = response.chat_history[-1]["content"]
    content = response.summary
    return {"messages": AIMessage(content=content, name="developer")}


def researcher_group_node(state: MessagesState):
    last_message = state["messages"][-1]
    response = research_team.invoke(last_message.content)  # type: ignore
    content = response.chat_history[-1]["content"]
    return {"messages": AIMessage(content=content, name="researcher")}


def expert_group_node(state: MessagesState):
    last_message = state["messages"][-1]
    response = expert_team.invoke(last_message.content)  # type: ignore
    content = response.chat_history[-1]["content"]
    return {"messages": AIMessage(content=content, name="expert")}


def ask_human_node(state: MessagesState):
    last_message = state["messages"][-1]
    # response = expert_team.invoke(last_message.content)  # type: ignore
    # content = response.chat_history[-1]["content"]
    # input_msg = last_message.content
    # resp = input(input_msg)
    input_msg = "Hello, I'm the code analysis assistant, please ask any questions about the project."
    return {"messages": AIMessage(content=input_msg, name="ask_user")}


builder = StateGraph(MessagesState)
builder.add_conditional_edges(START, route_node)
builder.add_node(code_analysis_team.name, code_analysis_group_node)
# builder.add_node(research_team.name, researcher_group_node)
# builder.add_node(expert_team.name, expert_group_node)
# builder.add_node("General", general_agent)
# builder.add_node("HumanInvolved", ask_human_node)

graph = builder.compile(debug=DEVELOPMENT)

graph.get_graph().draw_png("graph.png")
