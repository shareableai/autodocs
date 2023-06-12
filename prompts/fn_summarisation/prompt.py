from langchain.prompts import StringPromptTemplate
from pydantic import validator, BaseModel

from prompts.fn_summarisation.examples import FnSummarisationExampleSelector


class FunctionSummarisationPromptTemplate(StringPromptTemplate, BaseModel):
    _prompt = """
You are a Data Scientist summarising the following function code for a technical colleague.

{function}
 
Write a description of what the function does, focusing on elements that help explain the transformations the model 
applies to the input data, and how this is performed. 
For each argument to the function, describe the argument and how it affects the function output.

{examples}
Description:
"""

    _example_prompt = """
## Function
{function}

## Summary
{summary}
===
"""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 1 or "function" not in v:
            raise ValueError("function must be provided")
        return v

    def empty_prompt(self) -> str:
        return self.format("")

    def format(self, function: str) -> str:
        examples = FnSummarisationExampleSelector.examples(function)
        formatted_examples = [
            self._example_prompt.format(
                function=example["function"],
                summary=example["summary"],
            )
            for example in examples
        ]
        prompt = self._prompt.format(
            function=function,
            examples="",
        )
        return prompt

    def _prompt_type(self) -> str:
        return "function_summarizer"
