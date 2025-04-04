import logging
from time import time
from typing import Dict, Literal, Optional

import numpy as np
from llama_index.core import Response, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from trulens.apps.llamaindex import TruLlama
from trulens_eval import Tru
from trulens_eval.app import App
from trulens_eval.feedback import Feedback, GroundTruthAgreement

from index_service.custom_embedding import OllamaEmbedding
from index_service.custom_ollama import Ollama


class LlamaIndexOllamaEvaluator:
    def __init__(
        self,
        document_query_engine: BaseQueryEngine,
        source_code_query_engine: BaseQueryEngine,
        embedding_model: OllamaEmbedding,
    ):
        self.document_query_engine = document_query_engine
        self.source_code_query_engine = source_code_query_engine
        self.tru = Tru()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embed_model = embedding_model

        try:
            self.document_tru_recorder = TruLlama(
                self.document_query_engine, app_id="document_index"
            )
            self.source_code_tru_recorder = TruLlama(
                self.source_code_query_engine, app_id="source_code_index"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize TruLlama: {e}")
            self.document_tru_recorder = None
            self.source_code_tru_recorder = None

        # self.groundedness_metric = Feedback(GroundTruthAgreement())

    def query(
        self,
        query: str,
        data_type: Literal["document", "source_code"],
        evaluate: bool = False,
        expected_result: Optional[str] = None,
    ):
        query_engine = (
            self.document_query_engine
            if data_type == "document"
            else self.source_code_query_engine
        )

        try:
            results = query_engine.query(query)

            if not isinstance(results, Response):
                raise ValueError(
                    "Query engine response is not a valid Response object."
                )

            self.logger.info(f"Query executed successfully for: {query}")

            if evaluate and self.document_tru_recorder:
                tru_recorder = (
                    self.document_tru_recorder
                    if data_type == "document"
                    else self.source_code_tru_recorder
                )

                try:
                    context = TruLlama.select_context(query_engine)
                except Exception as e:
                    self.logger.warning(f"Failed to extract context: {e}")
                    context = None

                feedbacks = self.generate_feedbacks(query, results, expected_result)

                record = tru_recorder.record(
                    model=lambda x: results,
                    inputs={"query": query},
                    outputs={"result": results},
                    feedbacks=[self.groundedness_metric] + feedbacks,
                )
                self.logger.info(f"Evaluation completed for query: {query}")
                return results, record

            return results, None

        except Exception as e:
            self.logger.error(f"Error during query or evaluation: {e}")
            return None, None

    def generate_feedbacks(
        self, query: str, results: str, expected_result: Optional[str] = None
    ) -> list:
        feedbacks = []

        if expected_result:
            feedbacks.append(Feedback(reference=expected_result, output=results))

        start_time = time()
        end_time = time()
        response_time = end_time - start_time
        feedbacks.append(Feedback(metric="latency", value=response_time))

        return feedbacks

    def show_dashboard(self, port: int = 8502):
        try:
            self.tru.run_dashboard(port=port, force=True)
            self.logger.info(
                f"Dashboard launched successfully at http://localhost:{port}"
            )
        except Exception as e:
            self.logger.error(f"Error launching dashboard: {e}")
            raise
