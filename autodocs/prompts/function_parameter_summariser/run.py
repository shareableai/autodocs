import logging
from typing import Any

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel
from openai import InvalidRequestError

from autodocs.utils.model import ChatModel
from autodocs.prompts.function_parameter_summariser.prompt import (
    FunctionRefinePrompt,
    FunctionParameterPrompt,
)
from autodocs.utils.function_description import FunctionDescription
from autodocs.utils.token_splitter import gpt_token_splitter

LOGGER = logging.getLogger(__name__)


class FnParameterSummarizer:
    def __init__(self, model: BaseChatModel = ChatModel.load_model()):
        self.model = model
        self.question_prompt = PromptTemplate(
            template=FunctionParameterPrompt, input_variables=["text"]
        )
        self.refine_prompt = PromptTemplate(
            template=FunctionRefinePrompt,
            input_variables=["existing_answer", "text"],
        )
        self.called_fn_summarise_chain = load_summarize_chain(
            self.model,
            chain_type="refine",
            return_intermediate_steps=False,
            question_prompt=self.question_prompt,
            refine_prompt=self.refine_prompt,
        )

    def _split_arguments(self, function_source: str) -> Any:
        splitter = gpt_token_splitter()
        return splitter.create_documents(
            ["Function Source: " + "\n" + f"{function_source}"]
        )

    @classmethod
    def _format_output(cls, output_text: str) -> dict[str, str]:
        parameter_dict = {}
        parameters = output_text.split('\n')
        for parameter in parameters:
            if len(parameter.strip()) == 0:
                continue
            try:
                name, description = parameter.split(': ', 1)
                if ', ' in name:
                    names = name.split(', ')
                    for name in names:
                        parameter_dict[name] = description
                else:
                    parameter_dict[name] = description
            except ValueError:
                breakpoint()
        return parameter_dict

    def __call__(
        self, function: FunctionDescription
    ) -> dict[str, str]:
        split_arguments = self._split_arguments(function.source)
        LOGGER.info(
            "Summarising Function %s using %s requests.",
            function.name,
            len(split_arguments),
        )
        try:
            data_input = {
                "documentation": "", #function.caller_docs if function.caller_docs is not None else "No Documentation Provided\n",
                "input_documents": split_arguments
            }
            summaries = self.called_fn_summarise_chain(data_input)
            return self._format_output(summaries["output_text"])
        except InvalidRequestError:
            breakpoint()
