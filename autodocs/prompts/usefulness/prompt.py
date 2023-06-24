from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator


def format_response(response: str) -> list[tuple[str, float]]:
    responses = []
    for idx, item in enumerate(response.split("\n")):
        item = item.replace(f"{idx + 1}. ", "")
        fn_name, fn_value = item.split(" (")
        fn_value = float(fn_value.replace(")", ""))
        responses.append((fn_name, fn_value))
    return responses


class TraceImportancePrompt(StringPromptTemplate, BaseModel):
    """A custom prompt template that takes in source code and function calls as input, and identifies the relevant
    tags for the model"""

    _prompt = prompt_template = """
You are a Machine Learning Engineer reading a model's code for the first time. I will tell you which functions were used
in the model, and it is your job to direct me towards which functions will give you the most information about the purpose
of the model. Ignore validation functions.

Return up to the top 5 most important function calls, with a rating from 0.0 to 1.0 indicating how important it is that 
you learn more about that function to understand the model. 
A rating of 0 indicates it is unlikely to teach you something about how the model works. A rating of 1 indicates it is 
essential to understanding the model. A rating of 0.5 indicates you can likely guess what the function.

## Functions Executed
[0]     model_lib.__main__
[1]         model_lib.check_values_in_range
[1]         model_lib.preprocessing
[2]             model_lib.colourise_image
[1]         model_lib.fit

## Important Functions
1. model_lib.fit (1.0)
2. model_lib.preprocessing (1.0)
3. model_lib.__main__ (1.0)
4. model_lib.colourise_image (0.5)

## Functions Executed
{trace}

## Important Functions
"""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 1 or "trace" not in v:
            raise ValueError("trace must be provided")
        return v

    def format(self, trace: str) -> str:
        prompt = self._prompt.format(
            trace=trace,
        )
        return prompt

    def _prompt_type(self) -> str:
        return "tag-generator"
