import logging
import os
from typing import Literal

from autogen import GroupChatManager, UserProxyAgent
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_experimental.utilities import PythonREPL
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import TypedDict

from agent_service import constants, logger
from agent_service.utility import chat_llm
from shared.constants import *

memory = MemorySaver()


def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )


def make_supervisor_node(llm: BaseChatModel, members: list[str]):
    options = ["FINISH"] + members

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[tuple(options)]  # type: ignore

    supervisor_agent = create_react_agent(
        llm,
        tools=[],
        state_modifier=make_system_prompt(
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: {members}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status."
            " When finished, respond with FINISH."
        ),
    )

    def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:  # type: ignore
        """An LLM-based router."""
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: {members}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status."
            " When finished, respond with FINISH."
        )
        messages = [
            AIMessage(
                content=system_prompt,
                name="supervisor",
            )
        ] + state["messages"]

        if len(state["messages"]) < 1 or state["messages"][-1].name == "user":
            response = llm.with_structured_output(Router).invoke(messages)
            if not response:
                goto = "general"
            else:
                goto = response["next"]  # type: ignore
                if goto == "FINISH":
                    goto = END
        else:
            goto = END
        return Command(goto=goto)

    return supervisor_node


general_agent = create_react_agent(
    basic_llm,
    tools=[],
    state_modifier="""
You are a specialized assistant responsible for handling any questions or requests that are outside the scope of specific business workflows or domain-specific tasks. 
Your primary role is to:
1. Provide clear, accurate, and concise answers to general questions.
2. Handle unexpected or ambiguous queries by interpreting the user's intent to the best of your ability.
3. Route unresolved or unrelated requests to the appropriate agent or suggest alternative solutions.
4. Inform the user when their request cannot be fulfilled, along with a reason why and potential next steps.

Key Behaviors:
- If the user’s query seems unrelated to existing workflows or agents, clarify the intent or propose general solutions.
- Avoid providing incomplete or misleading information. Always base your responses on verified data or logic.
- If you cannot assist the user, explicitly communicate that, with empathy and professionalism.

Instructions:
- Always strive for a helpful and user-friendly tone in your responses.
- In cases of doubt, seek clarification from the user.
- Do not hesitate to acknowledge when a request is beyond the capabilities of the system.
""",
)


def troubleshooting_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = ragproxyagent.initiate_chat(
        developer,
        message=ragproxyagent.message_generator,
        problem=state,
        search_string="spark",
    )
    result["messages"][-1] = AIMessage(
        content=result["messages"][-1].content, name="troubleshooting"
    )
    return Command(
        update={"messages": result["messages"]},
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


def general_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = general_agent.invoke(state)

    result["messages"][-1] = AIMessage(
        content=result["messages"][-1].content, name="general"
    )
    return Command(
        update={"messages": result["messages"]},
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


def vector_node(state: MessagesState) -> Command[Literal["tree"]]:
    result = general_agent.invoke(state)

    result["messages"][-1] = AIMessage(
        content=result["messages"][-1].content, name="vector"
    )
    return Command(
        update={"messages": result["messages"]},
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="tree",
    )


def tree_node(state: MessagesState) -> Command[Literal["graph"]]:
    result = general_agent.invoke(state)

    result["messages"][-1] = AIMessage(
        content=result["messages"][-1].content, name="tree"
    )
    return Command(
        update={"messages": result["messages"]},
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="graph",
    )


def graph_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = general_agent.invoke(state)

    result["messages"][-1] = AIMessage(
        content=result["messages"][-1].content, name="graph"
    )
    return Command(
        update={"messages": result["messages"]},
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


supervisor_node = make_supervisor_node(basic_llm, ["general", "troubleshooting"])

workflow = StateGraph(MessagesState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("general", general_node)
workflow.add_node("troubleshooting", troubleshooting_node)

workflow.add_node("vector", vector_node)
workflow.add_node("tree", tree_node)
workflow.add_node("graph", graph_node)

workflow.add_edge("supervisor", "vector")
workflow.add_edge("vector", "tree")
workflow.add_edge("tree", "graph")

workflow.add_edge(START, "supervisor")

graph = workflow.compile(checkpointer=memory, debug=True)

graph.get_graph().draw_png("workflow.png")


def stream_graph_updates(user_input: str):
    for event in graph.stream(
        # {"messages": [("user", user_input)]}, config={"configurable": {"thread_id": 42}}
        {"messages": HumanMessage(content=user_input, name="user")},
        config={"configurable": {"thread_id": 42}},
    ):
        for value in event.values():
            # print("Assistant:", value["messages"][-1].content)
            if value:
                value["messages"][-1].pretty_print()


# while True:
#     message = input("User:")
#     stream_graph_updates(message)
