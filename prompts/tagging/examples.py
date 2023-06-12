import os
from typing import Optional, Any, List, Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma


class TagGenerationExampleSelector:
    _example_selector: Optional[SemanticSimilarityExampleSelector] = None

    @classmethod
    def _load_examples(cls) -> Optional[List[Dict]]:
        return [
            {
                "trace": """
    [1] 		__main__.Net.forward
    [2] 			torch.nn.functional.dropout
    [2] 			torch.nn.modules.linear.Linear.forward
    [2] 			torch.nn.functional.sigmoid""",
                "functions": """
    def __main__.Net.__init__(self):
        self.fc1 = nn.Linear(200, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def __main__.Net.forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))""",
                "tags": "classification: 1.0, regression: 0.1",
            }
        ]

    @classmethod
    def _setup(cls):
        if (examples := TagGenerationExampleSelector._load_examples()) is not None:
            TagGenerationExampleSelector._example_selector = (
                SemanticSimilarityExampleSelector.from_examples(
                    examples,
                    OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
                    Chroma,
                    k=1,
                )
            )

    @classmethod
    def examples(cls, trace: str, functions: str) -> Optional[Any]:
        if TagGenerationExampleSelector._example_selector is None:
            TagGenerationExampleSelector._setup()
        if (selector := TagGenerationExampleSelector._example_selector) is not None:
            return selector.select_examples({"trace": trace, "functions": functions})
