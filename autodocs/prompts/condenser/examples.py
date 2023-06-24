import json
import os
from typing import Any, Dict, List, Optional

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma


class CondenserExampleSelector:
    _example_selector: Optional[SemanticSimilarityExampleSelector] = None

    @classmethod
    def _load_examples(cls) -> Optional[List[Dict]]:
        with open("/prompts/condenser/data/data.json") as f:
            return json.load(f)

    @classmethod
    def _setup(cls):
        if (examples := CondenserExampleSelector._load_examples()) is not None:
            CondenserExampleSelector._example_selector = (
                SemanticSimilarityExampleSelector.from_examples(
                    examples,
                    OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
                    Chroma,
                    k=1,
                )
            )

    @classmethod
    def examples(cls, functions: str, trace: list[str]) -> Optional[Any]:
        if CondenserExampleSelector._example_selector is None:
            CondenserExampleSelector._setup()
        if (selector := CondenserExampleSelector._example_selector) is not None:
            return selector.select_examples({"functions": functions, "trace": trace})
