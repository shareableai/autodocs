import logging
from typing import Any, Iterator

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.text_splitter import TokenTextSplitter
from openai import InvalidRequestError

from autodocs.prompts.component_identifier.prompt import (
    ComponentIdentifierQAPrompt,
    ComponentsRefinePrompt,
)
from autodocs.utils.function_description import FunctionDescription

LOGGER = logging.getLogger(__name__)


class ComponentIdentifierQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.question_prompt = PromptTemplate(
            template=ComponentIdentifierQAPrompt, input_variables=["text", "trace_type"]
        )
        self.refine_prompt = PromptTemplate(
            template=ComponentsRefinePrompt,
            input_variables=["existing_answer", "text", "trace_type"],
        )
        self.component_summarise_chain = load_summarize_chain(
            self.model,
            chain_type="refine",
            return_intermediate_steps=False,
            question_prompt=self.question_prompt,
            refine_prompt=self.refine_prompt,
        )

    def _split_source(self, functions: list[tuple[FunctionDescription, str]]) -> Any:
        splitter = TokenTextSplitter(chunk_size=3_000, chunk_overlap=250)
        return splitter.create_documents(
            [
                f"Name: {function.name}" + "\n" + f"Description: {desc}" + "\n\n"
                for (function, desc) in functions
            ]
        )
    
    def _format_response(self, response: str) -> dict[str, str]:
        components: dict[str,str] = {}
        for line in response.split('\n'):
            if len(line.strip()) == 0:
                continue
            if ':' not in line:
                LOGGER.warn("Expected delimiter in %s - found none", line)
                continue
            component_name, component_description = line.split(':')
            component_name = component_name.removeprefix('(').removesuffix(')')
            components[component_name] = component_description
        return components


    def __call__(
        self, functions: list[tuple[FunctionDescription, str]], trace_type: str
    ) -> dict[str, str]:
        split_arguments = self._split_source(functions)
        LOGGER.info(
            "Identifying Components using %s requests.",
            len(split_arguments),
        )
        try:
            output_text = self.component_summarise_chain(
                {"input_documents": split_arguments, "trace_type": trace_type}
            )['output_text']
            breakpoint()
            return self._format_response(output_text)
        except InvalidRequestError as e:
            LOGGER.error("Invalid Request - %s", str(e))
            raise e
