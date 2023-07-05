from typing import Type
from uuid import UUID

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from autodocs.pipeline.run import Trace


class ListClassesToolInput(BaseModel):
    """Inputs for get_trace"""
    trace_id: str = Field(description="TraceID in UUID format")
    trace_type: str = Field(description="TraceType - either Inference, Preprocessing, or Train")


class ListClassesTool(BaseTool):
    name = "get_function_tool"
    description = """
        Useful when you want a list of classes tracked by a program
        You should enter the TraceID provided in the task as `trace_id`.
        You should also enter the Trace Type provided in the task as `trace_type`.
        """
    args_schema: Type[BaseModel] = ListClassesToolInput

    def _run(self, trace_id: str, trace_type: str):
        trace = Trace.from_id(UUID(trace_id), trace_type)
        class_dir = trace.root_dir.parent / 'classes'
        classes = [x.name.replace('.json', '') for x in class_dir.glob('*')]
        if len(classes) == 0:
            return "No classes in program"
        return '\n'.join(classes)

    def _arun(self, trace_id: str, trace_type: str, function_name: str):
        raise NotImplementedError("get_trace does not support async")


if __name__ == "__main__":
    from autodocs.pipeline.model import ChatModel
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent

    llm = ChatModel().load_model()
    tools = [
        ListClassesTool(),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)
    agent.run("""
    TraceID: 68f54d92-819f-428a-9b84-a0ad63c65fc2
    TraceType: inference

    Which classes were executed in the above program?
    """)