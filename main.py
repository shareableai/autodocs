import os

os.environ["OPENAI_API_KEY"] = "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"

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI


template = """
Rules
As a Machine Learning Engineer, you must assign a numerical score between 0 and 1 to reflect the similarity of two model summaries composed by your colleagues. 
A score of 1 is to be given when the summaries express the same concepts, but with different words. 
However, if the summaries are discussing different models, assign a score of 0. 
If it is indeterminable if the summaries are describing the same model, a score of 0 should be given.


As a Machine Learning Engineer, you must assign a numerical value between 0 and 1 to evaluate the similarity of two model summaries created by your colleagues. 
A score of 1 should be given when the summaries are discussing the same concept but using different words. 
On the other hand, if the summaries are describing different models, a score of 0 should be assigned. 
If it is not possible to determine if the summaries are discussing the same model, a score of 0 should be given.

Evaluate the similarity of two model summaries authored by your peers, assigning a score between 0 and 1. 
If the summaries are talking about the same model, assign 1. If the summaries are conveying distinct concepts, 
even if they share some words, assign 0. If you cannot determine if the summaries are discussing the same model, assign 0.


"""


if __name__ == "__main__":
    prompt = PromptTemplate(
        input_variables=["sentence_a", "sentence_b"],
        template=template,
    )
    model = ChatOpenAI(model_name="gpt-3.5-turbo")  # switch to 'gpt-4'
    qa = LLMChain(llm=model, prompt=prompt)
    print(
        qa.run(
            sentence_a="I walked to the apothecary",
            sentence_b="I strolled to the drugstore",
        )
    )
