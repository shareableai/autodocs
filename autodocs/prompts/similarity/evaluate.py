import json

import numpy as np
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from autodocs.prompts.similarity.prompt import OutputSimilarityPromptTemplate
from sklearn.metrics import mean_squared_error

from autodocs.prompts.similarity.utils import format_response


def evaluate_similarity(prompt: OutputSimilarityPromptTemplate) -> float:
    model = ChatOpenAI(model_name="gpt-3.5-turbo")  # switch to 'gpt-4'
    qa = LLMChain(llm=model, prompt=prompt)
    with open("/similarity/data/data.json") as f:
        all_examples = json.load(f)

    responses = []
    scores = []
    for example in all_examples:
        response = qa.run(
            summary_a=example["summary_a"], summary_b=example["summary_b"]
        )
        try:
            formatted_response = format_response(response)
        except Exception:
            print(f"Could not format prompt response {response}")
            raise RuntimeError
        responses.append(formatted_response)
        scores.append(example["similarity_score"])

    for example_idx, example in enumerate(all_examples):
        print(f"{example['summary_a']=}")
        print(f"{example['summary_b']=}")
        print(f"Predicted: {responses[example_idx]}, Actual: {scores[example_idx]}")

    return mean_squared_error(responses, scores)


def modify_rules(
    prompt: OutputSimilarityPromptTemplate,
) -> OutputSimilarityPromptTemplate:
    from langchain import PromptTemplate, OpenAI, LLMChain

    prompt_template = """
Generate a variation of the following instruction while keeping the semantic meaning.

{rules}
"""

    llm = OpenAI(temperature=0.5)
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    result = llm_chain(prompt.rules())
    prompt.set_rules(result["text"])
    return prompt


if __name__ == "__main__":
    prompt = OutputSimilarityPromptTemplate(input_variables=["summary_a", "summary_b"])
    score_prompt: dict[str, float] = {}
    for _ in range(10):
        previous_prompt = prompt.rules()
        modify_rules(prompt)
        try:
            similarity_score = evaluate_similarity(prompt)
            score_prompt[prompt.rules()] = similarity_score
        except Exception:
            print(f"Skipped prompt {prompt.rules()}")
            prompt.set_rules(previous_prompt)
    prompts = list(score_prompt.keys())
    scores = list(score_prompt.values())
    for best_score_index in np.argsort(scores).tolist():
        print(f"Score {scores[best_score_index]}, Prompt: {prompts[best_score_index]}")
