from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from autodocs.prompts.usefulness.prompt import TraceImportancePrompt, format_response


class ImportanceQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.prompt = TraceImportancePrompt(input_variables=["trace"])
        self.importance_qa = LLMChain(llm=self.model, prompt=self.prompt)

    def __call__(self, trace: list[str]) -> list[str]:
        raw_response = self.importance_qa.run("\n".join(trace))
        formatted_response = format_response(raw_response)
        return [res for (res, conf) in formatted_response if conf > 0.5]
