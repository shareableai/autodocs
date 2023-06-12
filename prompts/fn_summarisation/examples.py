import json
import os
from typing import Optional, Any, List, Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma


class FnSummarisationExampleSelector:
    _example_selector: Optional[SemanticSimilarityExampleSelector] = None

    @classmethod
    def _load_examples(cls) -> Optional[List[Dict]]:
        with open(
            "D:\ShareableAI/autodocs/prompts/fn_summarisation/data/data.json"
        ) as f:
            return json.load(f)

    @classmethod
    def _setup(cls):
        if (examples := FnSummarisationExampleSelector._load_examples()) is not None:
            FnSummarisationExampleSelector._example_selector = (
                SemanticSimilarityExampleSelector.from_examples(
                    examples,
                    OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
                    Chroma,
                    k=1,
                )
            )

    @classmethod
    def examples(cls, function: str) -> Optional[Any]:
        if FnSummarisationExampleSelector._example_selector is None:
            FnSummarisationExampleSelector._setup()
        if (selector := FnSummarisationExampleSelector._example_selector) is not None:
            return selector.select_examples({"function": function})
