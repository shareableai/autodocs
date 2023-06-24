from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator


class HyperparameterFinderPromptTemplate(StringPromptTemplate, BaseModel):
    _prompt = """
You are a Data Scientist identifying hyperparameters in the arguments provided to a function. You are looking to write down
the name of the hyperparameter, how the hyperparameter is referenced in the code, and why you know it is a hyperparameter.

A hyperparameter is considered to be a tuning parameter, something that can be configured by the Data Scientist.

Think step by step, making sure you're correct as to how you know an argument is a hyperparameter.

Provide the output in the format 

Hyperparameter Names, Paths, and Reasoning:
1. Hyperparameter Name [Path]: Reasoning
2. Hyperparameter Name [Path]: Reasoning
...

If you did not find any hyperparameters, instead write "I did not find any hyperparameters."

Function:
def transform(self, x: int) -> int:
    return int(self.transform_type(x))

The transform function applies the calllable transformation referenced in self.transform_type to the input x, returning an integer.

Hyperparameter Name, Paths, and Reasoning:
1. transform_type [self.transform_type]: This argument determines what type of transformation is applied to x, and can be configured during class instantiation.

Function:
def transform(cls, x: int) -> int:
    return int(cls.transform_type(x))

The transform function applies the calllable transformation referenced in cls.transform_type to the input x, returning an integer.

Hyperparameter Name, Paths, and Reasoning:
1. transform_type [cls.transform_type]: This argument determines what type of transformation is applied to x, and can be configured during class instantiation.


Function:
def apply(self, x: torch.Tensor, applications: int, method: str):
    if method == 'cvf':
        for _ in range(applications):
            x = torch_mod.cvf(x) + self.bias
    elif method == 'fgd':
        for _ in range(applications):
            x = torch_mod.fgd(x) + self.bias
    return x

The apply method applies one of a number of modifying functions to the input X, likely returning a Tensor.

Hyperparameter Name, Paths, and Reasoning:
1. method [local.method]: This argument determines what type of transformation is applied to x, and can be configured during class instantiation.
2. applications [local.applications]: This argument determines how many times the application is applied to x, would directly impact the performance of the model, and is easily configurable by the data scientist running the function.

Function:
{function_source}

{function_description}

Hyperparameter Names, Paths, and Reasoning:
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

    def format(self, function_source: str, function_description: str) -> str:
        prompt = self._prompt.format(
            function_source=function_source,
            function_description=function_description,
        )
        return prompt

    def _prompt_type(self) -> str:
        return "hyperparameter_identifier"
