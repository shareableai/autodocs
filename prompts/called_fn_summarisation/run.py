from typing import Iterator, Any

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from openai import InvalidRequestError

from langchain.text_splitter import TokenTextSplitter

from prompts.called_fn_summarisation.prompt import (
    CalledFunctionQuestionPrompt,
    CalledFunctionRefinePrompt,
)
from utils.function_description import FunctionDescription


class CalledFnSummarisationQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.question_prompt = PromptTemplate(
            template=CalledFunctionQuestionPrompt, input_variables=["text"]
        )
        self.refine_prompt = PromptTemplate(
            template=CalledFunctionRefinePrompt,
            input_variables=["existing_answer", "text"],
        )
        self.called_fn_summarise_chain = load_summarize_chain(
            self.model,
            chain_type="refine",
            return_intermediate_steps=False,
            question_prompt=self.question_prompt,
            refine_prompt=self.refine_prompt,
        )

    def _split_arguments(
        self, function_description: str, function_arguments: dict[str, str]
    ) -> Any:
        splitter = TokenTextSplitter(chunk_size=3_000, chunk_overlap=0)
        return splitter.create_documents(
            [
                f"Function Description: {function_description}\n"
                f"{['Argument: ' + arg_name + ' = ' + arg for (arg_name, arg) in function_arguments.items()]}"
            ]
        )

    def __call__(
        self, function_descriptions: list[tuple[FunctionDescription, str]]
    ) -> Iterator[tuple[FunctionDescription, str]]:
        for function, fn_summary in function_descriptions:
            split_arguments = self._split_arguments(fn_summary, function.arguments)
            print(
                f"Summarising Function {function.name} using {len(split_arguments)} requests."
            )
            try:
                data_input = {"input_documents": split_arguments}
                summaries = self.called_fn_summarise_chain(data_input)
                yield function, summaries["output_text"]
            except InvalidRequestError:
                breakpoint()
