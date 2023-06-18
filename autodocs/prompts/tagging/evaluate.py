import json
import re
from typing import Dict

import numpy as np
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from sklearn.metrics import roc_auc_score

from autodocs.prompts.tagging.prompt import TagGenerationPromptTemplate

RESPONSE_REGEX = re.compile(r"[0-9]{1,}\. ([a-z\-_]+): ([0-9].?[0-9]?)")


def format_response(response: str) -> Dict[str, float]:
    tags = {}
    for tag_line in response.split("\n"):
        matches = RESPONSE_REGEX.search(tag_line)
        tags[matches.group(1)] = float(matches.group(2))
    return tags


if __name__ == "__main__":
    prompt = TagGenerationPromptTemplate(input_variables=["trace", "functions"])
    model = ChatOpenAI(model_name="gpt-3.5-turbo")  # switch to 'gpt-4'
    qa = LLMChain(llm=model, prompt=prompt)
    with open("/tagging/data/data.json") as f:
        all_examples = json.load(f)

    scores = []

    for example in all_examples:
        response = qa.run(trace=example["trace"], functions=example["functions"])
        formatted_response = format_response(response)
        ground_truth = [
            1 if resp in example["tags"] else 0 for resp in formatted_response.keys()
        ]
        confidences = list(formatted_response.values())
        scores.append(roc_auc_score(ground_truth, confidences))

    print(scores)
    print(np.mean(scores))
