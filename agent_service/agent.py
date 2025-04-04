from typing import Annotated, Any, Callable, Dict, Literal, Optional, Union

from autogen import ConversableAgent, UserProxyAgent

from agent_service.tools import run_meta_prompting
from agent_service.utility import llm_config


class MetaAgent(ConversableAgent):
    SYSTEM_MESSAGE = """You are Meta-Expert, an extremely clever expert with the unique ability to collaborate with multiple experts (such as Expert Problem Solver, Expert Mathematician, Expert Essayist, etc.) to tackle any task and solve any complex problems. Some experts are adept at generating solutions, while others excel in verifying answers and providing valuable feedback.
As Meta-Expert, your role is to oversee the communication between the experts, effectively using their skills to answer a given question while applying your own critical thinking and verification abilities.
To communicate with a expert, call function "meta_prompting" with the expert's name, identity information and the task that needs to be solved. The function will return a response from the expert.
Ensure that your instructions are clear and unambiguous, and include all necessary information within the triple quotes. You should assign personas to the experts (e.g., "You are a physicist specialized in...").
You can interact with only one expert at a time, and break complex problems into smaller, solvable tasks. Each interaction is treated as an isolated event, so include all relevant details in every call.
Refrain from repeating the very same questions to experts. Examine their responses carefully and seek clarification if required, keeping in mind they don't recall past interactions.
Upon the completion of all tasks and verifications, you should conclude the result and reply "TERMINATE".
"""
    TOOL = {
        "type": "function",
        "function": {
            "name": "meta_prompting",
            "description": "Solve a task by querying an expert. Provide the expert identity and the task that needs to be solved, and the function will return the response of the expert.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "[REQUIRED] The task that needs to be solved by the expert.",
                    },
                    "expert_name": {
                        "type": "string",
                        "description": "[REQUIRED] Name of the expert. Should follow the format: Expert xxx.",
                    },
                    "expert_identity": {
                        "type": "string",
                        "description": "[REQUIRED] A high-quality description about the most capable and suitable expert to answer the instruction. In second person perspective. For example, You are a linguist, well-versed in the study of language and its structures. You are equipped with a good understanding of grammar rules and can differentiate between nouns, verbs, adjectives, adverbs, etc. You can quickly and accurately identify the parts of speech in a sentence and explain the role of each word in the sentence. Your expertise in language and grammar is highly valuable in analyzing and understanding the nuances of communication.",
                    },
                },
            },
        },
    }

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = None,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        code_execution_config: Union[dict, Literal[False]] = False,
        description: Optional[
            str
        ] = "A helpful AI assistant that can build a group of agents at a proper time to solve a task.",
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=self.SYSTEM_MESSAGE,
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            description=description,
            **kwargs,
        )
        self.update_tool_signature(self.TOOL, is_remove=False)
