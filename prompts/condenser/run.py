from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from prompts.condenser.prompt import CondenserPromptTemplate
from utils.function_description import FunctionDescription


class CondenserQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.prompt = CondenserPromptTemplate(
            input_variables=["functions", "trace", "trace_type"]
        )
        self.condense_qa = LLMChain(llm=self.model, prompt=self.prompt)

    def __call__(
        self,
        functions: list[tuple[FunctionDescription, str]],
        trace: list[str],
        trace_type: str,
    ) -> str:
        return self.condense_qa.run(
            functions=functions, trace=trace, trace_type=trace_type
        )


if __name__ == "__main__":
    trace = (
        [
            "__main__.<module>",
            "sklearn.svm._base.fit",
            "sklearn.base.get_params",
            "sklearn.base._get_param_names",
            " sklearn.base.<listcomp>",
            " sklearn.base.<listcomp>",
            "scipy.sparse._base.isspmatrix",
        ],
    )
    fn_input = {
        "sklearn.svm._base.fit": "Fit a SVM model with various options including the choice of kernel and sample "
        "weights. The function takes in training data (X and y) and an optional array of "
        "sample weights. The model can be fit using either sparse or dense data and the "
        "function handles input validation to ensure compatibility between input shapes. The "
        "specific kernel used can be set as an argument to the function, with RBF being the "
        "default. The function also allows for setting a random seed for reproducibility.",
        "sklearn.base.get_params": "This function returns the parameters used in the estimator, including any "
        "sub-estimators within it. The output is a dictionary with parameter names mapped "
        "to their values. The argument 'deep' is set to True by default, which means it "
        "will also return the parameters for any sub-estimators.",
    }
    # TODO: Add Trace to Condenser
    print(CondenserPromptTemplate.format_functions(fn_input, trace))
    # print(CondenserQA()(fn_input))
