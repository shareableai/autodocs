from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from prompts.similarity.prompt import OutputSimilarityPromptTemplate
from prompts.similarity.utils import format_response


def calculate_similarity(model: ChatOpenAI, summary_a: str, summary_b: str) -> float:
    prompt = OutputSimilarityPromptTemplate(input_variables=["summary_a", "summary_b"])
    qa = LLMChain(llm=model, prompt=prompt)
    response = qa.run(summary_a=summary_a, summary_b=summary_b)
    return format_response(response)
