import logging
from typing import Any

import cachetools

from autodocs.prompts.function_hyperparameter_summariser.run import FnHyperparameterSummarizer
from autodocs.prompts.function_parameter_summariser.run import FnParameterSummarizer

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
    logging.info("Hyperparameter Description: %s", desc)
    return desc


def retrieve_parameter_descriptions(
        function_description: FunctionDescription,
) -> dict[str, str]:
    """Returns the Parameter Name and Description."""
    desc = FnParameterSummarizer()(function_description)
    logging.info("Function Parameter Description: %s", desc)
    return desc


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
