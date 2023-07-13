import logging
from typing import Any

import cachetools

from autodocs.prompts.filter_description.run import FilterQA
from autodocs.prompts.function_hyperparameter_summariser.run import FnHyperparameterSummarizer

from autodocs.utils.function_description import FunctionDescription


def _retrieve_function_parameters(
    function_description: FunctionDescription, parameter_names: list[str]
) -> list[Any]:
    return [
        function_description.retrieve_property(parameter.strip())
        for parameter in parameter_names
    ]


def _retrieve_hyperparameter_names_descriptions(
    function_description: FunctionDescription,
) -> dict[str, str]:
    """Returns the Hyperparameter Name and reason for selection in a dictionary."""
    _, desc = next(iter(FnHyperparameterSummarizer()([function_description])))
    logging.info("Description: %s", desc)
    filtered_desc = FilterQA()(
        desc,
        "hyperparameters within the description",
        "a list of those hyperparameters, i.e. ['a', 'b']",
        "If there are no hyperparameters, return an empty list"
    )
    logging.info("Filtered Description %s", filtered_desc)
    try:
        filtered_desc_list = eval(filtered_desc)
        if not isinstance(filtered_desc_list, list):
            raise RuntimeError
        filtered_desc_dict = {}
        if len(filtered_desc_list) > 0:
            for function_parameter in filtered_desc_list:
                if function_parameter.strip() != '**kwargs':
                    function_parameter_summary = FilterQA()(
                        desc,
                        f"original reasoning for whether {function_parameter} is a hyperparameter",
                        "a short sentence that keeps the original meaning of the description",
                        ""
                    )
                    filtered_desc_dict[function_parameter] = function_parameter_summary
        return filtered_desc_dict
    except RuntimeError:
        return {}


def _retrieve_parameter_descriptions(
    function_description: FunctionDescription,
) -> dict[str, str]:
    """Returns the Parameter Name and Description."""
    _, desc = next(iter(FnHyperparameterSummarizer()([function_description])))
    breakpoint()
    logging.info("Description: %s", desc)
    filtered_desc = FilterQA()(
        desc,
        "parameters that are hyperparameters within the description provided",
        "a list of those hyperparameters, i.e. ['a', 'b']",
        "If there are no hyperparameters, return an empty list."
    )
    logging.info("Filtered Description %s", filtered_desc)
    try:
        filtered_desc_list = eval(filtered_desc)
        if not isinstance(filtered_desc_list, list):
            raise RuntimeError
        filtered_desc_dict = {}
        if len(filtered_desc_list) > 0:
            for function_parameter in filtered_desc_list:
                if function_parameter.strip() != '**kwargs':
                    function_parameter_summary = FilterQA()(
                        desc,
                        f"original reasoning for whether {function_parameter} is a hyperparameter",
                        "a short sentence that keeps the original meaning of the description",
                        ""
                    )
                    filtered_desc_dict[function_parameter] = function_parameter_summary
        return filtered_desc_dict
    except RuntimeError:
        return {}


@cachetools.cached(cache=cachetools.LRUCache(maxsize=1))
def function_hyperparameters(
    function_description: FunctionDescription,
) -> dict[str, tuple[Any, str]]:
    hyperparameter_names: dict[str, str] = _retrieve_hyperparameter_names_descriptions(
        function_description
    )
    hyperparameter_values: list[Any] = _retrieve_function_parameters(
        function_description, list(hyperparameter_names.keys())
    )
    return {
        parameter_name: (parameter_value, parameter_description)
        for ((parameter_name, parameter_description), parameter_value) in zip(
            hyperparameter_names.items(), hyperparameter_values
        )
    }
