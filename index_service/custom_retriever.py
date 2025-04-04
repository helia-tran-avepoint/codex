import re
from typing import Any, List

from llama_index.core.indices import property_graph

pattern = r"<think>.*</think>"


class LLMSynonymRetriever(property_graph.LLMSynonymRetriever):
    def _parse_llm_output(self, output: str) -> List[str]:
        if self._output_parsing_fn:
            matches = self._output_parsing_fn(output)
        else:
            matches = output.strip().split("^")

        # capitalize to normalize with ingestion
        clean_matches = []
        for x in matches:
            text = re.sub(pattern, "", x, flags=re.DOTALL).strip()
            if text:
                clean_matches.append(text)
        return clean_matches
