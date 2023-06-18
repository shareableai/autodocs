from langchain.prompts import StringPromptTemplate
from pydantic import validator, BaseModel


class HyperparameterFinderPromptTemplate(StringPromptTemplate, BaseModel):
    _prompt = """
You are a Data Scientist identifying hyperparameters in the arguments provided to a function.

A hyperparameter is considered to be a tuning parameter, something that can be configured by the Data Scientist.

Think step by step, providing the full name of the hyperparameter as well as why you believe it to be a hyperparameter.

Write each hyperparameter exactly as it is found within the code. 

Provide the output in the format 

Hyperparameter Names and Justifications:
1. Hyperparameter Name: Justification
2. Hyperparameter Name: Justification
...

If you did not find any hyperparameters, instead write "I did not find any hyperparameters."

Function:
def transform(self, x: int)
The transform function applies the calllable transformation referenced in self.transform_type to the input x, returning an integer.

Hyperparameter Names and Justifications:
1. self.transform_type: This argument determines what type of transformation is applied to x, and can be configured during class instantiation.

Function:
{function_signature}
{function_description}

Hyperparameter Names and Justifications:
"""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 3 or "function_description" not in v:
            raise ValueError("function must be provided")
        return v

    @staticmethod
    def _format_arguments(arguments: dict[str, str]) -> str:
        return "\n".join(
            f"{arg_name}: {arg_value}" for (arg_name, arg_value) in arguments.items()
        )

    def format(
        self, function_arguments: dict[str, str], function_signature: str, function_description: str
    ) -> str:
        prompt = self._prompt.format(
            # function_arguments=self._format_arguments(function_arguments),
            function_signature=function_signature,
            function_description=function_description,
            fn_argument_names=', '.join(function_arguments.keys())
        )
        return prompt

    def _prompt_type(self) -> str:
        return "hyperparameter_identifier"
