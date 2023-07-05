from langchain.agents import AgentType
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent
from autodocs.agent_tools.parameter_retrieval import ParameterValueGetter
from autodocs.agent_tools.trace_comparisons import TraceComparator
from autodocs.agent_tools.trace_retrieval import TraceGetter
from autodocs.agents.system_message import MODEL_COMPARISON_SYSTEM
from autodocs.pipeline.model import ChatModel

if __name__ == "__main__":
    llm = ChatModel.load_model()
    agent = initialize_agent(
        tools=[ParameterValueGetter(), TraceGetter(), TraceComparator()],
        llm=llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, 
        agent_kwargs={
            "system_message": SystemMessage(content=MODEL_COMPARISON_SYSTEM)
        },
    )

    agent.run("""
    Program A:
    TraceID: 68f54d92-819f-428a-9b84-a0ad63c65fc2
    TraceType: Inference

    Program B:
    TraceID: 2d0fa205-a8f4-4695-ac4c-79e73da08d6f
    TraceType: Inference
    
    Program C:
    TraceID: 2d0fa205-a8f4-4695-ac4c-79e73da08d6f
    TraceType: Inference
    
    What is the difference between Program A Program B and Program C?
    """)
