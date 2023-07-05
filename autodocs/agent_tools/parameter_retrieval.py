from langchain.agents import AgentType
from langchain.agents import initialize_agent
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from autodocs.utils.function_description import load_tracked_class_property


class GetParameterInput(BaseModel):
    """Inputs for get_parameter_value"""
    class_id: str = Field(description="Class Identifier in UUID format")
    trace_id: str = Field(description="TraceID in UUID format")
    parameter_name: str = Field(description="Name of the Parameter")


class ParameterValueGetter(BaseTool):
    name = "get_parameter_value"
    description = """
        Useful when you want to get the value or code of a class parameter, including class methods.
        You should enter the class ID provided in the task as `class_id`.
        You should also enter the TraceID provided in the task as `trace_id`.
        You should also enter the parameter name as `parameter_name`
        """
    args_schema: Type[BaseModel] = GetParameterInput

    def _run(self, class_id: str, trace_id: str, parameter_name: str):
        param_response = load_tracked_class_property(class_id, trace_id, parameter_name)
        if param_response is None:
            return "Can't access that parameter."
        return param_response

    def _arun(self, class_id: str, trace_id: str, parameter_name: str):
        raise NotImplementedError("get_parameter_value does not support async")


if __name__ == "__main__":
    from autodocs.pipeline.model import ChatModel

    llm = ChatModel().load_model()
    tools = [
        ParameterValueGetter(),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("""
    TraceID: 83265c6b-f74e-4db4-9acb-c662563b4c2a
    ClassID: 7d1dc378-d59a-4170-9d7e-a2e398bcc4bb
    
    Identify the hyperparameters in the method 'preprocess' from the above class.
    """)
