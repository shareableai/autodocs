from __future__ import annotations

import json
import logging
import pathlib
from pydoc import doc
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from autodocs.document.doc import OutputDocumentation
from autodocs.pipeline.model import ChatModel
from autodocs.prompts.called_fn_summarisation.run import CalledFnSummarisationQA
from autodocs.prompts.component_identifier.run import ComponentIdentifierQA
from autodocs.prompts.condenser.run import CondenserQA
from autodocs.prompts.hyperparameters.run import HyperparameterQA
from autodocs.prompts.usefulness.run import ImportanceQA
from autodocs.utils.function_description import FunctionDescription
from autodocs.utils.slugify import slugify

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def root_path() -> pathlib.Path:
    return pathlib.Path.home() / ".stack_traces"


class NoSuchTrace(BaseException):
    pass


@dataclass
class Trace:
    root_dir: pathlib.Path
    trace: list[str]
    trace_fns: list[str]
    trace_type: str

    @staticmethod
    def from_id(trace_id: uuid.UUID, trace_type: str) -> Trace:
        directory = root_path() / str(trace_id) / f"TrackingType.{trace_type}"
        if not directory.exists():
            raise NoSuchTrace(f"Could not find {trace_id}")
        with open(directory / "trace.txt") as f:
            trace = f.readlines()
            trace_fns = [
                fn_desc[4:].replace("\t", "").replace(" ", "").replace("\n", "")
                for fn_desc in trace
            ]
        return Trace(directory, trace, trace_fns, trace_type)

    def load_function_info(self, function_name: str) -> FunctionDescription:
        try:
            with open(self.root_dir / slugify(function_name)) as f:
                function_info = json.load(f)
                arguments: dict[str, str] = function_info.get("arguments", {})
                source: str = function_info.get("source", "")
                docs: str = function_info.get("caller_docs", "")
                caller_name: str = function_info.get("caller_name", "")
                signature: Optional[str] = function_info.get("signature", None)
                tracked_class_name: Optional[str] = function_info.get("tracked_class_name", None)
                return FunctionDescription(
                    function_name, source, docs, arguments, signature, self.root_dir, caller_name, tracked_class_name
                )
        except FileNotFoundError:
            return FunctionDescription(function_name, "", "", {}, None, self.root_dir, None, None)

    def _load_important_function_calls(self) -> list[str]:
        return ImportanceQA(model=ChatModel.model())(self.trace)

    def _retrieve_function_descriptions(
        self, important_functions: list[str]
    ) -> list[FunctionDescription]:
        return [
            self.load_function_info(function_call_name)
            for function_call_name in important_functions
        ]

    @staticmethod
    def _summarise_functions_with_arguments(
        function_descriptions: list[FunctionDescription],
    ) -> list[tuple[FunctionDescription, str]]:
        return list(
            CalledFnSummarisationQA(model=ChatModel.model())(
                functions=function_descriptions
            )
        )

    def _condense_descriptions(
        self, function_summaries: list[tuple[FunctionDescription, str]]
    ) -> str:
        return CondenserQA(model=ChatModel.model())(
            functions=function_summaries,
            trace=self.trace_fns,
            trace_type=self.trace_type,
        )

    def _find_components(
        self, function_summaries: list[tuple[FunctionDescription, str]]
    ) -> str:
        return ComponentIdentifierQA(model=ChatModel.model())(
            functions=function_summaries,
            trace_type=self.trace_type,
        )

    def _find_hyperparameters(
        self, function_summaries: list[tuple[FunctionDescription, str]]
    ) -> list[tuple[FunctionDescription, dict[str, Any]]]:
        return list(HyperparameterQA(model=ChatModel.model())(function_summaries))

    def _function_summaries(self) -> list[tuple[FunctionDescription, str]]:
        # Identify usefulness of each component within the trace
        important_functions = self._load_important_function_calls()
        LOGGER.info(
            "Identified useful functions as; %s", ", ".join(important_functions)
        )
        # Retrieve Source info for each of the important functions from the dump
        function_descriptions = self._retrieve_function_descriptions(
            important_functions
        )
        LOGGER.info("Retrieved Function info from observer")
        # Summarise the Function Source Info
        function_summaries = self._summarise_functions_with_arguments(
            function_descriptions
        )
        LOGGER.info(
            "Generated Function Summaries for %s functions", len(function_summaries)
        )
        return function_summaries

    def create_description(self) -> tuple[str, dict[str, Any]]:
        function_summaries = self._function_summaries()
        #   TODO - Extend this using saved items - frequently it's not possible to identify the objects referred to because
        #       they existed on an object, not on the function. Having a saved item via Jackdaw would make this accessible again.
        # Identify hyperparameters within the functions, based on the descriptions of the functions.
        hyperparameters = self._find_hyperparameters(function_summaries)
        components = self._find_components(function_summaries)
        return components, hyperparameters


def load_library_versions(observation_id: uuid.UUID) -> dict[str, str]:
    version_path = root_path() / str(observation_id) / "versions.txt"
    if not version_path.exists():
        raise NoSuchTrace
    with open(version_path) as f:
        return json.load(f)


def create_output_document(observation_id: uuid.UUID) -> OutputDocumentation:
    try:
        processing_description, processing_hyperparams = Trace.from_id(
            observation_id, "Processing"
        ).create_description()
    except NoSuchTrace:
        processing_hyperparams = {}
        processing_description = None
    try:
        training_description, training_hyperparams = Trace.from_id(
            observation_id, "Training"
        ).create_description()
    except NoSuchTrace:
        training_hyperparams = {}
        training_description = None
    try:
        inference_description, inference_hyperparams = Trace.from_id(
            observation_id, "Inference"
        ).create_description()
    except NoSuchTrace:
        inference_hyperparams = {}
        inference_description = None
    breakpoint()
    return OutputDocumentation(
        preprocessing_steps=processing_description,
        training_steps=training_description,
        inference_steps=inference_description,
        parameters = processing_hyperparams | training_hyperparams | inference_hyperparams,
        total_libraries=load_library_versions(observation_id),
    )


if __name__ == "__main__":
    document = create_output_document("04d024f0-22c2-4433-8dfe-6c024a245737")
    print(document.steps())
    print(document.parameters)
