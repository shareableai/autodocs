from langchain.prompts import StringPromptTemplate
from pydantic import validator, BaseModel

from utils.function_description import FunctionDescription


class CondenserPromptTemplate(StringPromptTemplate, BaseModel):
    _prompt = """
You are a Data Scientist describing the transformations that are applied to input data by a machine learning
model during the {trace_type} stage.
Given the following function descriptions, describe each transformation that happens to the input data, keeping a 
focus on the input data at each step. 

{examples}

# Functions
{functions}

# Steps
"""

    _example_prompt = """
    ## Function
    {functions}

    ## Steps
    {steps}
    ===
    """

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if (
            len(v) != 3
            or "functions" not in v
            or "trace" not in v
            or "trace_type" not in v
        ):
            raise ValueError("functions and trace must be provided")
        return v

    @staticmethod
    def format_functions(
        functions: dict[str, [tuple[FunctionDescription, str]]], trace: list[str]
    ) -> str:
        prompt = ""
        for idx, function_name in enumerate(trace):
            if function_name not in functions:
                prompt = f"{prompt}\n{idx}. {function_name}"
            else:
                fn_desc, summary = functions[function_name]
                prompt = f"{prompt}\n{idx}. {function_name}: {summary}"

        return prompt

    def format(
        self,
        functions: list[tuple[FunctionDescription, str]],
        trace: list[str],
        trace_type: str,
    ) -> str:
        """examples = CondenserExampleSelector.examples(functions, trace)
        formatted_examples = [
            self._example_prompt.format(
                function=example["functions"],
                summary=example["steps"],
            )
            for example in examples
        ]"""
        prompt = self._prompt.format(
            functions=self.format_functions(
                {fn.name: (fn, summary) for (fn, summary) in functions}, trace
            ),
            trace_type=trace_type.lower(),
            examples="",
        )
        return prompt

    def _prompt_type(self) -> str:
        return "condense_summaries"
