from re import S
from typing import Iterator, Any, Iterable

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from autodocs.prompts.hyperparameters.prompt import HyperparameterFinderPromptTemplate
from autodocs.utils.function_description import FunctionDescription


class HyperparameterQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.prompt = HyperparameterFinderPromptTemplate(
            input_variables=["function_arguments", "function_description", "function_signature"]
        )
        self.hyperparameter_qa = LLMChain(llm=self.model, prompt=self.prompt)

    @staticmethod
    def _format_response(
        response: str, fn: FunctionDescription
    ) -> Iterable[tuple[str, Any]]:
        for line in response.split("\n"):
            if (len(line.strip()) > 0
                and "I did not find any hyperparam" not in line 
                and any(str(x) in line for x in range(10))
            ):
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
            hyperparameter_response = self.hyperparameter_qa.run(
                function_arguments=function.arguments,
                function_signature=function.signature if function.signature is not None else "",
                function_description=function_description,
            )
            yield function, dict(
                self._format_response(hyperparameter_response, function)
            )
