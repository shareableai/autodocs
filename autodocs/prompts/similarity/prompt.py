from langchain.prompts import StringPromptTemplate
from pydantic import validator, BaseModel

from autodocs.prompts.similarity.examples import SimilarityExampleSelector


class OutputSimilarityPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template that takes in source code and function calls as input, and provides a summary of actions taken in the code"""

    _rules = """Create a score ranging from 0 to 1 to evaluate the similarity between the two model summaries created 
    by your colleagues. Assign a 1 if the summaries are referring to the same model; if the summaries are different, 
    even if they have some of the same words, assign a 0. If it is unclear if they are discussing the same model, 
    assign a 0.5."""
    _prompt = """
{rules}

===
{examples}

## First Summary
{summary_a}

## Second Summary
{summary_b}

## Similarity
"""

    _example_prompt = """
## First Summary
{summary_a}

## Second Summary
{summary_b}

## Similarity
{similarity_score}
===
"""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 2 or "summary_a" not in v or "summary_b" not in v:
            raise ValueError("summary_a and summary_b must be provided")
        return v

    @classmethod
    def rules(cls) -> str:
        return cls._rules

    @classmethod
    def set_rules(cls, new_rules: str) -> None:
        cls._rules = new_rules

    def format(self, summary_a: str, summary_b: str) -> str:
        examples = SimilarityExampleSelector.examples(summary_a, summary_b)
        formatted_examples = [
            self._example_prompt.format(
                summary_a=example["summary_a"],
                summary_b=example["summary_b"],
                similarity_score=example["similarity_score"],
            )
            for example in examples
        ]
        prompt = self._prompt.format(
            rules=self.rules(),
            summary_a=summary_a,
            summary_b=summary_b,
            examples="".join(formatted_examples),
        )
        return prompt

    def _prompt_type(self) -> str:
        return "summary_similarity"
