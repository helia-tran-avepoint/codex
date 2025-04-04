import re
from typing import Tuple

from autogen.formatting_utils import colored
from autogen.io.websockets import IOStream
from llama_index.core.agent.react.output_parser import (
    ReActOutputParser,
    extract_final_response,
    parse_action_reasoning_step,
)
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.output_parsers.utils import extract_json_str
from llama_index.core.types import BaseOutputParser


class CustomOutputParser(ReActOutputParser):
    """ReAct Output parser."""

    def parse(self, output: str, is_streaming: bool = False) -> BaseReasoningStep:
        """Parse output from ReAct agent.

        We expect the output to be in one of the following formats:
        1. If the agent need to use a tool to answer the question:
            ```
            Thought: <thought>
            Action: <action>
            Action Input: <action_input>
            ```
        2. If the agent can answer the question without any tools:
            ```
            Thought: <thought>
            Answer: <answer>
            ```
        """
        iostream = IOStream.get_default()

        if "Thought:" not in output:
            # NOTE: handle the case where the agent directly outputs the answer
            # instead of following the thought-answer format
            response = ResponseReasoningStep(
                thought="(Implicit) I can answer without any more tools!",
                response=output,
                is_streaming=is_streaming,
            )

        # An "Action" should take priority over an "Answer"
        elif "Action:" in output:
            response = parse_action_reasoning_step(output)

        elif "Answer:" in output:
            thought, answer = extract_final_response(output)
            response = ResponseReasoningStep(
                thought=thought, response=answer, is_streaming=is_streaming
            )

        iostream.print(colored(response.get_content() + "\n", "blue"))
        return response
        # raise ValueError(f"Could not parse output: {output}")

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError
