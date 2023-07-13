import logging
from typing import Any, Iterator

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel
from langchain.text_splitter import TokenTextSplitter
from openai import InvalidRequestError

from autodocs.utils.model import ChatModel
from autodocs.prompts.function_hyperparameter_summariser.prompt import (
    FunctionRefinePrompt,
    FunctionParameterPrompt,
)
from autodocs.utils.function_description import FunctionDescription

LOGGER = logging.getLogger(__name__)


class FnHyperparameterSummarizer:
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
        splitter = TokenTextSplitter(chunk_size=3_000, chunk_overlap=250)
        return splitter.create_documents(
            ["Function Source: " + "\n" + f"{function_source}"]
        )

    def __call__(
        self, functions: list[FunctionDescription]
    ) -> Iterator[tuple[FunctionDescription, str]]:
        for function in functions:
            split_arguments = self._split_arguments(function.source)
            LOGGER.info(
                "Summarising Function %s using %s requests.",
                function.name,
                len(split_arguments),
            )
            try:
                data_input = {"input_documents": split_arguments}
                summaries = self.called_fn_summarise_chain(data_input)
                yield function, summaries["output_text"]
            except InvalidRequestError:
                breakpoint()
