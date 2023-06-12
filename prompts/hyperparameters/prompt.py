from langchain.prompts import StringPromptTemplate
from pydantic import validator, BaseModel


class HyperparameterFinderPromptTemplate(StringPromptTemplate, BaseModel):
    _prompt = """
You are a Data Scientist identifying hyperparameters in the arguments provided to the following function:

{function_description}

Think step by step, providing the hyperparameter as well as your justification for why each argument is a hyperparameter. 
Write each hyperparameter on a separate line. 

If a hyperparameter is on the `self` object, write the full path to that parameter, i.e. `self.x` rather than `x`.

Provide the output in the format 

Hyperparameter Names and Justifications
1. Hyperparameter Name: Justification
2. Hyperparameter Name: Justification
...

If there are no hyperparameters, write "I did not find any hyperparameters."

Hyperparameter Names and Justifications
"""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 2 or "function_description" not in v:
            raise ValueError("function must be provided")
        return v

    @staticmethod
    def _format_arguments(arguments: dict[str, str]) -> str:
        return "\n".join(
            f"{arg_name}: {arg_value}" for (arg_name, arg_value) in arguments.items()
        )

    def format(
        self, function_arguments: dict[str, str], function_description: str
    ) -> str:
        prompt = self._prompt.format(
            # function_arguments=self._format_arguments(function_arguments),
            function_description=function_description,
        )
        return prompt

    def _prompt_type(self) -> str:
        return "hyperparameter_identifier"
