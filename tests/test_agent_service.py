import os
import sys

from langchain_core.messages import HumanMessage

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from agent_service import app_config
from agent_service.graph import graph
from agent_service.startup import generate_service_port

agent_port, index_port, web_port, analysis_port = generate_service_port("agent_service")
app_config.index_service_port = index_port
app_config.analysis_service_port = analysis_port


def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": HumanMessage(content=user_input, name="user")},
        config={"configurable": {"thread_id": 42}},
    ):
        for value in event.values():
            if value:
                if isinstance(value["messages"], list):
                    value["messages"][-1].pretty_print()
                else:
                    value["messages"].pretty_print()


while True:
    message = input("User:")
    stream_graph_updates(message)
