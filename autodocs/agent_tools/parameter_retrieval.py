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
        Useful when you want to get the value of a class parameter.
        You should enter the class ID provided in the task.
        You should also enter the TraceID provided in the task.
        You should also enter the parameter name
        """
    args_schema: Type[BaseModel] = GetParameterInput

    def _run(self, class_id: str, trace_id: str, parameter_name: str):
        price_response = load_tracked_class_property(class_id, trace_id, parameter_name)
        return price_response

    def _arun(self, class_id: str, trace_id: str, parameter_name: str):
        raise NotImplementedError("get_parameter_value does not support async")


if __name__ == "__main__":
    from autodocs.pipeline.model import ChatModel
    llm = ChatModel().load_model()
    tools = [
        ParameterValueGetter(),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)
    agent.run("""
    TraceID: cf56f7f5-6567-4db2-8999-34dfad25d071
    ClassID: 6d20cf97-06fa-442e-9f13-b5a36a634236
    
    What is the value of the parameter image_mean on the class?
    """)