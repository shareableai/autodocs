from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from autodocs.pipeline.run import Trace


class GetTraceInput(BaseModel):
    """Inputs for get_trace"""
    trace_id: str = Field(description="TraceID in UUID format")
    trace_type: str = Field(description="TraceType - either Inference, Preprocessing, or Train")

class TraceGetter(BaseTool):
    name = "get_trace"
    description = """
        Useful when you want to get a list of executed functions from a program.
        You should enter the TraceID provided in the task as `trace_id`.
        You should also enter the TraceType provided in the task as `trace_type`.
        """
    args_schema: Type[BaseModel] = GetTraceInput

    def _run(self, trace_id: str, trace_type: str):
        trace = Trace.from_id(trace_id, trace_type)
        return '\n'.join(trace.trace_fns)

    def _arun(self, trace_id: str, trace_type: str):
        raise NotImplementedError("get_trace does not support async")


if __name__ == "__main__":
    from autodocs.pipeline.model import ChatModel
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent
    llm = ChatModel().load_model()
    tools = [
        TraceGetter(),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)
    agent.run("""
    TraceID: 68f54d92-819f-428a-9b84-a0ad63c65fc2
    TraceType: inference
    
    What functions were executed in the program?
    """)