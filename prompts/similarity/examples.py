import os
from typing import Optional, Any, List, Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma


class SimilarityExampleSelector:
    _example_selector: Optional[SemanticSimilarityExampleSelector] = None

    @classmethod
    def _load_examples(cls) -> Optional[List[Dict]]:
        return [
            {
                "summary_a": """
1. Load image from file
2. Remove artifacts via neural network
3. Detect bounding boxes via CRAFT
4. Detect text from each bounding box
                """,
                "summary_b": """
1. Load input image data
2. Clean up image data - removing artifacts
3. Identify bounding boxes
4. Run text detection
                """,
                "similarity_score": "1",
            },
            {
                "summary_a": """
1. Load tabular data with Pandas
2. Normalize all continuous variables
3. Remove all NAs
4. Use LightGBM Classifier to predict likelihood of cancer
                        """,
                "summary_b": """
1. Load tabular data
2. Preprocess input data
3. Classify data
                        """,
                "similarity_score": "0",
            },
            {
                "summary_a": """
1. Load tabular data with Pandas
2. Normalize all continuous variables
3. Remove all NAs
4. Use LightGBM Classifier to predict likelihood of cancer
                                """,
                "summary_b": """
1. Load input data via Pandas
2. Preprocess input data, normalizing numeric columns and removing NAs
3. Classify using Decision Trees
                                """,
                "similarity_score": "1",
            },
        ]

    @classmethod
    def _setup(cls):
        if (examples := SimilarityExampleSelector._load_examples()) is not None:
            SimilarityExampleSelector._example_selector = (
                SemanticSimilarityExampleSelector.from_examples(
                    examples,
                    OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
                    Chroma,
                    k=3,
                )
            )

    @classmethod
    def examples(cls, summary_a: str, summary_b: str) -> Optional[Any]:
        if SimilarityExampleSelector._example_selector is None:
            SimilarityExampleSelector._setup()
        if (selector := SimilarityExampleSelector._example_selector) is not None:
            return selector.select_examples(
                {"summary_a": summary_a, "summary_b": summary_b}
            )
