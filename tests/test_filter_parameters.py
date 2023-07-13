import pytest
from langchain.chat_models import ChatOpenAI

from autodocs.prompts.filter_description.run import FilterQA

INPUT_PROMPT = """
Description: 

- `self`: This is a reference to the instance of the class. It gives access to the attributes and methods within the class.

- `self.C`: This is the regularization parameter in the SVM. It is a hyperparameter that determines the trade-off between achieving a low training error and a low testing error.

- `self.max_iter`: This is the hard limit on iterations within solver, or -1 for no limit.

- `X`: This is the input data, which can be array-like, sparse matrix.

- `kernel`: This is the kernel type to be used in the algorithm.
"""
EXPECTED_OUTPUT = ["kernel", "self.C", "self.max_iter"]

FILTER_CONDITION = "hyperparameters"
OUTPUT_CONDITION = (
    "a list of hyperparameters, i.e. ['a', 'b'], written identically to how they were written in the "
    "task"
)


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-3.5-turbo",
    ],
)
def test_filter(model_name: str) -> None:
    prompt_runner = FilterQA(model=ChatOpenAI(model_name=model_name))
    try:
        output_text: str = prompt_runner.__call__(
            text_input=INPUT_PROMPT,
            condition=FILTER_CONDITION,
            output_format=OUTPUT_CONDITION,
        )
        formatted_output_text = eval(output_text)
    except SyntaxError:
        raise RuntimeError("%s could not be formatted into a list", output_text)
    assert isinstance(formatted_output_text, list)
    for expected_item in EXPECTED_OUTPUT:
        assert (
            expected_item in formatted_output_text
        ), f"{expected_item} not in {formatted_output_text}"
    for output_item in formatted_output_text:
        assert output_item in EXPECTED_OUTPUT, f"{output_item} not in {EXPECTED_OUTPUT}"
