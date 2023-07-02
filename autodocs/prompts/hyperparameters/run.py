import logging
import re
from typing import Any, Iterator

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from autodocs.pipeline.model import ChatModel

from autodocs.prompts.hyperparameters.prompt import HyperparameterFinderPromptTemplate
from autodocs.utils.slugify import slugify
from autodocs.utils.function_description import FunctionDescription

response_re = re.compile(r"[0-9]{1,}\.\s([^\s]+)\s\[([^:]+)\]:\s(.*)")

# TODO: This is a long prompt - change it to a summarisation-type prompt to work over a lot more data.

LOGGER = logging.getLogger(__name__)


class HyperparameterQA:
    def __init__(self, model: BaseChatModel = ChatOpenAI(model_name="gpt-3.5-turbo")):
        self.model = model
        self.prompt = HyperparameterFinderPromptTemplate(
            input_variables=["function_source", "function_description"]
        )
        self.hyperparameter_qa = LLMChain(llm=self.model, prompt=self.prompt)

    @staticmethod
    def _format_response(response: str, fn: FunctionDescription) -> dict[str, Any]:
        argument_locations = []
        arguments = {}
        for line in response.split("\n"):
            if (
                len(line.strip()) > 0
                and "I did not find any hyperparam" not in line
                and any(
                    str(x) in line for x in range(10)
                )  # Hyperparam format always has a number in it
            ):
                grouped_response = response_re.search(line)
                if grouped_response is None:
                    LOGGER.error(
                        "Failed to convert %s into hyperparameter format", line
                    )
                    breakpoint()
                (argument_name, argument_location, justification) = (
                    grouped_response.group(1),
                    grouped_response.group(2),
                    grouped_response.group(3),
                )
                argument_name = argument_name.replace("`", "")
                argument_locations.append(argument_location)
        for argument_location in argument_locations:
            if "local" in argument_location:
                norm_argument = argument_location.replace("local.", "")
                arguments[norm_argument] = fn.arguments.get(norm_argument, None)
            elif 'self' in argument_location or 'cls' in argument_location:
                argument_value = fn.load_class_property(argument_location)
                argument_location = f"{fn.caller_name}.{argument_location}"
                arguments[argument_location] = argument_value
        return arguments

    def __call__(
        self, functions: list[tuple[FunctionDescription, str]]
    ) -> Iterator[dict[str, Any]]:
        for function, function_description in functions:
            hyperparameter_response = self.hyperparameter_qa.run(
                function_source=function.source,
                function_description=function_description,
            )
            yield self._format_response(hyperparameter_response, function)


if __name__ == "__main__":
    import pathlib
    description = " Performs the forward pass of a model for object detection. It takes several arguments including pixel_values, pixel_mask, decoder_attention_mask, encoder_outputs, inputs_embeds, decoder_inputs_embeds, labels, output_attentions, output_hidden_states, and return_dict. The pixel_values are passed through a base model consisting of convolutional and batch normalization layers to obtain encoder and decoder outputs. The encoder outputs are further processed using self-attention mechanisms. The model applies a classifier to the sequence_output to obtain class logits and predicted bounding boxes. If labels are provided, it creates a matcher and a criterion, and computes losses based on the outputs and labels using smooth L1 loss for bounding box regression and cross-entropy loss for classification. The function returns the loss, loss_dict, logits, pred_boxes, auxiliary_outputs, and various hidden states and attentions depending on the return_dict parameter. "
    fn_desc = FunctionDescription.from_file(pathlib.Path.home() / '.stack_traces' / 'cf56f7f5-6567-4db2-8999-34dfad25d071' / 'TrackingType.Inference', slugify('transformers.models.detr.modeling_detr.forward'))
    descs = HyperparameterQA(ChatModel.model())([(fn_desc, description)])
    print(list(descs))
