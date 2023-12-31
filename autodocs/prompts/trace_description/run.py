import logging
from typing import Any, Iterator

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel
from langchain.text_splitter import TokenTextSplitter
from openai import InvalidRequestError

from autodocs.utils.model import ChatModel
from autodocs.prompts.trace_description.prompt import (
    TracePrompt,
    TraceRefinePrompt,
)

LOGGER = logging.getLogger(__name__)


class TraceQA:
    def __init__(self, model: BaseChatModel = ChatModel.load_model()):
        self.model = model
        self.question_prompt = PromptTemplate(
            template=TracePrompt, input_variables=["text", "trace_type"]
        )
        self.refine_prompt = PromptTemplate(
            template=TraceRefinePrompt,
            input_variables=["existing_answer", "text", "trace_type"],
        )
        self.called_fn_summarise_chain = load_summarize_chain(
            self.model,
            chain_type="refine",
            return_intermediate_steps=False,
            question_prompt=self.question_prompt,
            refine_prompt=self.refine_prompt,
        )

    @staticmethod
    def _split_trace(trace_description: str) -> Any:
        splitter = TokenTextSplitter(chunk_size=3_000, chunk_overlap=250)
        return splitter.create_documents([trace_description])

    def __call__(self, trace: str, trace_type: str) -> str:
        split_trace = self._split_trace(trace)
        LOGGER.info(
            "Describing Trace using %s requests.",
            len(split_trace),
        )
        try:
            data_input = {"input_documents": split_trace, "trace_type": trace_type}
            summaries = self.called_fn_summarise_chain(data_input)
            return summaries["output_text"]
        except InvalidRequestError:
            breakpoint()
