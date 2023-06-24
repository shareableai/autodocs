import logging
import re
from typing import Any, Iterator

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from autodocs.prompts.hyperparameters.prompt import HyperparameterFinderPromptTemplate
from autodocs.utils.function_description import FunctionDescription

response_re = re.compile(r"[0-9]{1,}\.\s([^\s]+)\s\[([^:]+)\]:\s(.*)")

# TODO: This is a long prompt - change it to a summarisation-type prompt to work over a lot more data.

LOGGER = logging.getLogger(__name__)


class HyperparameterQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.prompt = HyperparameterFinderPromptTemplate(
            input_variables=["function_source", "function_description"]
        )
        self.hyperparameter_qa = LLMChain(llm=self.model, prompt=self.prompt)

    @staticmethod
    def _format_response(response: str, fn: FunctionDescription) -> dict[str, Any]:
        argument_locations = []
        arguments = {}
        for line in response.split("\n"):
            if (
                len(line.strip()) > 0
                and "I did not find any hyperparam" not in line
                and any(
                    str(x) in line for x in range(10)
                )  # Hyperparam format always has a number in it
            ):
                grouped_response = response_re.search(line)
                if grouped_response is None:
                    LOGGER.error(
                        "Failed to convert %s into hyperparameter format", line
                    )
                    breakpoint()
                (argument_name, argument_location, justification) = (
                    grouped_response.group(1),
                    grouped_response.group(2),
                    grouped_response.group(3),
                )
                argument_name = argument_name.replace("`", "")
                argument_locations.append(argument_location)
        for argument_location in argument_locations:
            if "local" in argument_location:
                norm_argument = argument_location.replace("local.", "")
                arguments[norm_argument] = fn.arguments.get(norm_argument, None)
            elif 'self' in argument_location or 'cls' in argument_location:
                arg_loc = argument_location.removeprefix('self.').removeprefix('cls.')
                argument_value = fn.load_tracked_class_property(arg_loc)
                arg_loc = f"{fn.caller_name}.{arg_loc}"
                arguments[arg_loc] = argument_value
        return arguments

    def __call__(
        self, functions: list[tuple[FunctionDescription, str]]
    ) -> Iterator[tuple[FunctionDescription, dict[str, Any]]]:
        for function, function_description in functions:
            hyperparameter_response = self.hyperparameter_qa.run(
                function_source=function.source,
                function_description=function_description,
            )
            yield function, self._format_response(hyperparameter_response, function)
