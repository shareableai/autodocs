import difflib
from typing import Type
from uuid import UUID

from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from autodocs.pipeline.run import Trace


class CompareTraceInput(BaseModel):
    """Inputs for compare_traces"""
    trace_id: str = Field(description="TraceID in UUID format")
    trace_id_to_compare: str = Field(description="TraceID to compare in UUID format")
    trace_type: str = Field(description="TraceType - either Inference, Preprocessing, or Train")


class TraceComparator(BaseTool):
    name = "compare_traces"
    description = """
        Useful when you want to compare two Traces.
        You should enter a TraceID provided in the task as `trace_id`.
        You should also enter the TraceID to compare against as `trace_id_to_compare`
        You should also enter the TraceType provided in the task as `trace_type`
        
        Returns a delta of the Trace and the trace to compare.
        """
    args_schema: Type[BaseModel] = CompareTraceInput

    def _run(self, trace_id: str, trace_id_to_compare: str, trace_type: str):
        trace = Trace.from_id(UUID(trace_id), trace_type)
        rhs_trace = Trace.from_id(UUID(trace_id_to_compare), trace_type)
        trace_diff = '\n'.join(difflib.unified_diff(trace.trace_fns, rhs_trace.trace_fns))
        if len(trace_diff.strip()) == 0:
            return "Traces are identical"
        return trace_diff

    def _arun(self, trace_id: str, trace_id_to_compare: str, trace_type: str):
        trace = Trace.from_id(UUID(trace_id), trace_type)
        rhs_trace = Trace.from_id(UUID(trace_id_to_compare), trace_type)
        return '\n'.join(difflib.unified_diff(trace.trace_fns, rhs_trace.trace_fns))


if __name__ == "__main__":
    print(TraceComparator()._run(
        "68f54d92-819f-428a-9b84-a0ad63c65fc2",
        "5c6b1b62-69bc-4a20-a017-6406b66ec5bc",
        "Inference"
    ))
