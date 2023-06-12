from typing import Iterator

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from openai import InvalidRequestError

from prompts.fn_summarisation.prompt import FunctionSummarisationPromptTemplate
from utils.function_description import FunctionDescription


class FnSummarisationQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.prompt = FunctionSummarisationPromptTemplate(input_variables=["function"])
        self.summarise_qa = LLMChain(llm=self.model, prompt=self.prompt)

    def __call__(
        self, functions: list[FunctionDescription]
    ) -> Iterator[tuple[FunctionDescription, str]]:
        for function in functions:
            try:
                yield function, self.summarise_qa.run(function=function.source)
            except InvalidRequestError:
                breakpoint()
