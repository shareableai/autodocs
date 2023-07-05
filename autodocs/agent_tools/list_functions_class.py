import json
from typing import Type
from uuid import UUID

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from autodocs.pipeline.run import Trace


class ListFunctionByClassToolInput(BaseModel):
    """Inputs for get_trace"""
    trace_id: str = Field(description="TraceID in UUID format")
    trace_type: str = Field(description="TraceType - either Inference, Preprocessing, or Train")
    class_id: str = Field(description="Class ID")


class ListFunctionByClassTool(BaseTool):
    name = "get_functions_by_class_tool"
    description = """
        Useful when you want a list of functions that referenced a specific class in a trace.
        You should enter the TraceID provided in the task as `trace_id`.
        You should also enter the Trace Type provided in the task as `trace_type`.
        You should also enter the class id provided as `class_id`.
        """
    args_schema: Type[BaseModel] = ListFunctionByClassToolInput

    def _run(self, trace_id: str, trace_type: str, class_id: str):
        trace = Trace.from_id(UUID(trace_id), trace_type)
        affected_fns: list[str] = []
        for function_name in list(trace.root_dir.glob('*')):
            if function_name.name == 'trace.txt':
                continue
            fn_item = json.load(open(function_name, 'r'))
            if class_id in fn_item['tracked_argument_ids'].values():
                affected_fns.append(function_name.stem)

        if len(affected_fns) == 0:
            return "No functions referenced that class"
        return '\n'.join(affected_fns)

    def _arun(self, trace_id: str, trace_type: str, class_name: str):
        raise NotImplementedError("get_trace does not support async")


if __name__ == "__main__":
    from autodocs.pipeline.model import ChatModel
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent

    llm = ChatModel().load_model()
    tools = [
        ListFunctionByClassTool(),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)
    agent.run("""
    TraceID: 39c18290-4ed9-4c57-a3a6-49bde38725a8
    TraceType: inference
    ClassID: 5c226dfc-8f5b-4652-981c-86f531aa6318

    Which functions referenced the above class?
    """)