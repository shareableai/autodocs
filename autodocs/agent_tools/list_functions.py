from typing import Type
from uuid import UUID

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from autodocs.pipeline.run import Trace


class ListFunctionToolInput(BaseModel):
    """Inputs for get_trace"""
    trace_id: str = Field(description="TraceID in UUID format")
    trace_type: str = Field(description="TraceType - either Inference, Preprocessing, or Train")


class ListFunctionTool(BaseTool):
    name = "get_function_tool"
    description = """
        Useful when you want a list of functions executed by a program
        You should enter the TraceID provided in the task as `trace_id`.
        You should also enter the Trace Type provided in the task as `trace_type`.
        """
    args_schema: Type[BaseModel] = ListFunctionToolInput

    def _run(self, trace_id: str, trace_type: str):
        trace = Trace.from_id(UUID(trace_id), trace_type)
        return '\n'.join(trace.trace_fns)

    def _arun(self, trace_id: str, trace_type: str, function_name: str):
        raise NotImplementedError("get_trace does not support async")
