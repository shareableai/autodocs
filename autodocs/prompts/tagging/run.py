import re
from operator import gt
from typing import Any, Iterable, Iterator

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from autodocs.prompts.hyperparameters.prompt import HyperparameterFinderPromptTemplate
from autodocs.prompts.tagging.prompt import TagGenerationPromptTemplate
from autodocs.utils.function_description import FunctionDescription

response_re = re.compile(r"[0-9]{1,}\.\s([^\s]+)\s\[([^:]+)\]:\s(.*)")

raise NotImplementedError
# TODO: Reimplement using Function Descriptions.


class TagQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.prompt = TagGenerationPromptTemplate(input_variables=["trace", "function"])
        self.tag_qa = LLMChain(llm=self.model, prompt=self.prompt)

    def __call__(self, trace: str, functions) -> list[str]:
        for function, function_description in functions:
            tag_response = self.hyperparameter_qa.run(
                function_source=function.source,
                function_description=function_description,
            )
            yield function, self._format_response(hyperparameter_response, function)
