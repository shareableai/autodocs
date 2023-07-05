from typing import Type

from pydantic import BaseModel, Field
from langchain.tools import BaseTool


from autodocs.prompts.component_describer.run import ComponentDescriberQA


class GenerateDescriptionInput(BaseModel):
    """Inputs for get_description"""
    function_src: str = Field(description="Source code for a function")


class GenerateDescription(BaseTool):
    name = "get_description"
    description = """
        Useful when you want an explanation of a parameter or function source code
        You should enter the item you're interested in as `function_src`
    """
    args_schema: Type[BaseModel] = GenerateDescriptionInput

    def _run(self, function_src: str):
        return ComponentDescriberQA()(function_src)

    def _arun(self, function_src: str):
        return ComponentDescriberQA()(function_src)
