from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from autodocs.prompts.condenser.prompt import CondenserPrompt, CondenserRefinerPrompt
from autodocs.utils.function_description import FunctionDescription


class CondenserQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.question_prompt = PromptTemplate(
            template=CondenserPrompt, input_variables=["trace_type", "text"]
        )
        self.refine_prompt = PromptTemplate(
            template=CondenserRefinerPrompt,
            input_variables=["trace_type", "existing_answer", "text"],
        )
        self.condense_chain = load_summarize_chain(
            self.model,
            chain_type="refine",
            return_intermediate_steps=False,
            question_prompt=self.question_prompt,
            refine_prompt=self.refine_prompt,
        )

    def __call__(
        self,
        functions: list[tuple[FunctionDescription, str]],
        trace: list[str],
        trace_type: str,
    ) -> str:
        return self.condense_chain.run(
            functions=functions, trace=trace, trace_type=trace_type
        )
