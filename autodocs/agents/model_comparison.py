from langchain.agents import AgentType
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent
from autodocs.agent_tools.parameter_retrieval import ParameterValueGetter
from autodocs.agent_tools.trace_retrieval import GetTraceInput
from autodocs.agents.system_message import MODEL_COMPARISON_SYSTEM
from autodocs.pipeline.model import ChatModel


if __name__ == "__main__":
    llm = ChatModel.load_model()
    agent = initialize_agent(
        tools=[ParameterValueGetter(), GetTraceInput()],
        llm=llm, 
        agent=AgentType.OPENAI_FUNCTIONS, 
        verbose=True, 
        agent_kwargs={
            "system_message": SystemMessage(content=MODEL_COMPARISON_SYSTEM)
        },
    )

    agent.run("""
    Program A:
    TraceID: cf56f7f5-6567-4db2-8999-34dfad25d071
    TraceType: Inference

    Program B:
    TraceID: cf56f7f5-6567-4db2-8999-34dfad25d071
    TraceType: Inference
    
    What is the difference between Program A and Program B?
    """)
