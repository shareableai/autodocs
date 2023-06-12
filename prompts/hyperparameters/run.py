from typing import Iterator, Any, Iterable

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from openai import InvalidRequestError

from prompts.hyperparameters.prompt import HyperparameterFinderPromptTemplate
from utils.function_description import FunctionDescription


class HyperparameterQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.prompt = HyperparameterFinderPromptTemplate(
            input_variables=["function_arguments", "function_description"]
        )
        self.hyperparameter_qa = LLMChain(llm=self.model, prompt=self.prompt)

    @staticmethod
    def _format_response(
        response: str, fn: FunctionDescription
    ) -> Iterable[tuple[str, Any]]:
        for idx, line in enumerate(
            [line for line in response.split("\n") if len(line.strip()) > 0]
        ):
            if f"{idx + 1}" not in line:
                breakpoint()
            line = line.replace(f"{idx + 1}. ", "")
            (argument_name, justification) = line.split(": ")
            argument_name = argument_name.replace("`", "")
            if argument_name in fn.arguments:
                yield argument_name, fn.arguments[argument_name]
            else:
                pass

    def __call__(
        self, functions: list[tuple[FunctionDescription, str]]
    ) -> Iterator[tuple[FunctionDescription, dict[str, Any]]]:
        for function, function_description in functions:
            try:
                hyperparameter_response = self.hyperparameter_qa.run(
                    function_arguments=function.arguments,
                    function_description=function_description,
                )
                breakpoint()
                yield function, dict(
                    self._format_response(hyperparameter_response, function)
                )
            except InvalidRequestError:
                breakpoint()
