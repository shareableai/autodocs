import logging
from typing import Any

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.text_splitter import TokenTextSplitter
from openai import InvalidRequestError

from autodocs.prompts.component_describer.prompt import (
    ComponentDescriberQAPrompt,
    ComponentDescriberRefinePrompt,
)

LOGGER = logging.getLogger(__name__)


class ComponentDescriberQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.question_prompt = PromptTemplate(
            template=ComponentDescriberQAPrompt, input_variables=["text"]
        )
        self.refine_prompt = PromptTemplate(
            template=ComponentDescriberRefinePrompt,
            input_variables=["existing_answer", "text"],
        )
        self.component_summarise_chain = load_summarize_chain(
            self.model,
            chain_type="refine",
            return_intermediate_steps=False,
            question_prompt=self.question_prompt,
            refine_prompt=self.refine_prompt,
        )

    def _split_source(self, item: str) -> Any:
        splitter = TokenTextSplitter(chunk_size=3_000, chunk_overlap=250)
        return splitter.create_documents(
            [
                item
            ]
        )

    def __call__(
            self, item: str
    ) -> str:
        split_arguments = self._split_source(item)
        LOGGER.info(
            "Identifying Components using %s requests.",
            len(split_arguments),
        )
        try:
            output_text = self.component_summarise_chain(
                {"input_documents": item}
            )['output_text']
            return output_text
        except InvalidRequestError as e:
            LOGGER.error("Invalid Request - %s", str(e))
            raise e
