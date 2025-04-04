import os
from typing import Annotated
from urllib.parse import urljoin

import requests
from autogen import AssistantAgent, UserProxyAgent
from autogen.browser_utils.playwright_markdown_browser import PlaywrightMarkdownBrowser
from autogen.coding import DockerCommandLineCodeExecutor
from tavily import TavilyClient

from agent_service import app_config
from agent_service.utility import llm_config
from shared import constants
from shared.utils import DEVELOPMENT

tavily = TavilyClient(api_key=app_config.tavily_api_key)

# code_executor = DockerCommandLineCodeExecutor(
#     image="excutor_with_proxy",
#     timeout=300,
#     work_dir="docker_excutor_work_dir",
#     auto_remove=True,
# )


def run_meta_prompting(expert_name: str, expert_identity: str, task: str) -> str:
    """
    Run Meta-prompting to solve the task.
    The method is adapted from "Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding".
    Paper available at https://arxiv.org/abs/2401.12954
    """
    print("Running meta prompting...")
    print("Querying expert: ", expert_name)

    expert = AssistantAgent(
        name=expert_name,
        human_input_mode="NEVER",
        llm_config=llm_config,
        system_message='You are an AI assistant that helps people find information. Please answer the following question. Once you have determined the final answer, please present it using the format below:\n\n>> FINAL ANSWER:\n"""\n[final answer]\n"""',
        max_consecutive_auto_reply=10,
    )

    user_proxy = UserProxyAgent(
        name="proxy",
        human_input_mode="NEVER",
        default_auto_reply="TERMINATE",
        code_execution_config={"executor": code_executor},
        max_consecutive_auto_reply=10,
    )
    task += "\nYou have access to python code interpreter. Suggest python code block starting with '```python' and the code will be automatically executed. You can use code to solve the task or for result verification. You should always use print statement to get the value of a variable."
    user_proxy.initiate_chat(
        expert, message=expert_identity + "\n" + task, silent=DEVELOPMENT
    )

    expert_reply = user_proxy.chat_messages[expert][1]["content"]
    proxy_reply = user_proxy.chat_messages[expert][2]["content"]

    if proxy_reply != "TERMINATE":
        code_result = proxy_reply[
            proxy_reply.find("Code output:") + len("Code output:") :
        ].strip()
        expert_reply += (
            f"\nThis is the output of the code blocks when executed:\n{code_result}"
        )
    else:
        expert_reply.replace(
            "FINAL ANSWER:",
            f"{expert_name}'s final answer:\n",
        )

    return expert_reply


def search_tool(
    query: Annotated[str, "The search query"]
) -> Annotated[str, "The search results"]:
    return tavily.get_search_context(query=query, search_depth="advanced")


def link_visit_tool(link: str) -> str:
    """
    Visit and extract content by using PlaywrightMarkdownBrowser
    """
    browser = PlaywrightMarkdownBrowser()
    try:
        content = browser.visit_page(link)
        return content
    except Exception as e:
        return f"Error accessing link {link}: {e}"
    finally:
        browser.close()


def retrieve_knowledgebase(query: str) -> str:
    """
    Used to retrieve local document and get the project's knowledgebase.
    """
    try:
        payload = {
            "query": query,
            "data_type": "document",
            "evaluate": False,
        }
        response = requests.post(
            urljoin(
                f"http://{constants.INDEX_SERVICE_URL}:{app_config.index_service_port}",
                "retrieve",
            ),
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error processing the query: {str(e)}"


def retrieve_sourcecode(query: str) -> str:
    """
    Used to retrieve the project's sourcecode.
    """
    try:
        payload = {
            "query": query,
            "data_type": "source_code",
            "evaluate": False,
        }
        response = requests.post(
            urljoin(
                f"http://{constants.INDEX_SERVICE_URL}:{app_config.index_service_port}",
                "retrieve",
            ),
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error processing the query: {str(e)}"
