from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from autodocs.prompts.condenser.prompt import CondenserPromptTemplate
from autodocs.utils.function_description import FunctionDescription


class CondenserQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.prompt = CondenserPromptTemplate(
            input_variables=["functions", "trace", "trace_type"]
        )
        self.condense_qa = LLMChain(llm=self.model, prompt=self.prompt)

    def __call__(
        self,
        functions: list[tuple[FunctionDescription, str]],
        trace: list[str],
        trace_type: str,
    ) -> str:
        return self.condense_qa.run(
            functions=functions, trace=trace, trace_type=trace_type
        )