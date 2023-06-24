import json
import re
from typing import Dict

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from autodocs.prompts.usefulness.prompt import TraceImportancePrompt

RESPONSE_REGEX = re.compile(r"[0-9]{1,}\. ([a-z\-_]+): ([0-9].?[0-9]?)")


def format_response(response: str) -> Dict[str, float]:
    tags = {}
    for tag_line in response.split("\n"):
        matches = RESPONSE_REGEX.search(tag_line)
        tags[matches.group(1)] = float(matches.group(2))
    return tags


if __name__ == "__main__":
    prompt = TraceImportancePrompt(input_variables=["trace"])
    model = ChatOpenAI(model_name="gpt-3.5-turbo")  # switch to 'gpt-4'
    qa = LLMChain(llm=model, prompt=prompt)
    with open("/prompts/usefulness/data/data.json") as f:
        all_examples = json.load(f)

    for example in all_examples:
        response = qa.run(trace=example["trace"])
        print(response)
