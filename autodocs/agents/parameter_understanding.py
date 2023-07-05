from langchain.agents import AgentType
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent

from autodocs.agent_tools.describer import GenerateDescription
from autodocs.agent_tools.parameter_retrieval import ParameterValueGetter
from autodocs.pipeline.model import ChatModel

if __name__ == "__main__":
    llm = ChatModel.load_model()
    """Tools required
        [✔] Understand item from a class or function code
        [ ] 
        [✔] Retrieve parameter of a given attribute for a class
    """
    agent = initialize_agent(
        tools=[
            GenerateDescription(),
            ParameterValueGetter()
        ],
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    agent.run("""
    TraceID: 39c18290-4ed9-4c57-a3a6-49bde38725a8
    TraceType: inference
    ClassID: 5c226dfc-8f5b-4652-981c-86f531aa6318
    FunctionName: transformersimage_processing_utils_all 

    Find the hyperparameters of the named function.
    """)
