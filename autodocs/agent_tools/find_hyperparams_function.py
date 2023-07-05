from typing import Type
from uuid import UUID

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from autodocs.agent_tools.list_functions_class import ListFunctionByClassTool
from autodocs.pipeline.run import Trace
from autodocs.prompts.called_fn_summarisation.run import CalledFnSummarisationQA


class FunctionParamExplanationInput(BaseModel):
    """Inputs for get_trace"""
    trace_id: str = Field(description="TraceID in UUID format")
    trace_type: str = Field(description="TraceType - either Inference, Preprocessing, or Train")
    function_name: str = Field(description="Function Name")


class FunctionParamExplanation(BaseTool):
    name = "get_function_param_explanation"
    description = """
        Useful when you want an explanation of what the parameters in a function do.
        You should enter the TraceID provided in the task as `trace_id`.
        You should also enter the Trace Type provided in the task as `trace_type`.
        You should also enter the function name you're interested in as `function_name`.
        """
    args_schema: Type[BaseModel] = FunctionParamExplanationInput

    def _run(self, trace_id: str, trace_type: str, function_name: str):
        trace = Trace.from_id(UUID(trace_id), trace_type)
        function_desc = trace.load_function_info(function_name)
        # TODO Change to a Parameter-focused QA prompt
        _, desc = next(iter(CalledFnSummarisationQA(include_arguments=False)([function_desc])))
        return desc

    def _arun(self, trace_id: str, trace_type: str, function_name: str):
        raise NotImplementedError("get_trace does not support async")


if __name__ == "__main__":
    from autodocs.pipeline.model import ChatModel
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent

    llm = ChatModel().load_model()
    tools = [
        FunctionParamExplanation(),
        ListFunctionByClassTool(),
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    agent.run("""
    TraceID: 39c18290-4ed9-4c57-a3a6-49bde38725a8
    TraceType: inference
    ClassID: 5c226dfc-8f5b-4652-981c-86f531aa6318
    
    For each function that affects the above class, retrieve the hyperparameters for that function.
    """)