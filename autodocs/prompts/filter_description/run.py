import logging
from typing import Any

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel
from langchain.text_splitter import TokenTextSplitter

from autodocs.utils.model import ChatModel
from autodocs.prompts.filter_description.prompt import (
    FilterPrompt,
    FilterRefinePrompt,
)

LOGGER = logging.getLogger(__name__)


class FilterQA:
    def __init__(self, model: BaseChatModel = ChatModel.load_3_5_model()):
        self.model = model
        self.question_prompt = PromptTemplate(
            template=FilterPrompt,
            input_variables=["text", "condition", "output_format", "extra_conditions"],
        )
        self.refine_prompt = PromptTemplate(
            template=FilterRefinePrompt,
            input_variables=["existing_answer", "text", "condition", "output_format", "extra_conditions"],
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

    def __call__(self, text_input: str, condition: str, output_format: str, extra_conditions: str) -> str:
        splitter = TokenTextSplitter(chunk_size=3_000, chunk_overlap=250)
        split_text = splitter.create_documents([text_input])
        LOGGER.info(
            "Filtering using %s requests.",
            len(split_text),
        )
        data_input = {
            "input_documents": split_text,
            "condition": condition,
            "output_format": output_format,
            "extra_conditions": extra_conditions
        }
        summaries = self.called_fn_summarise_chain(data_input)
        return summaries["output_text"]
