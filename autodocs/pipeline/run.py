from __future__ import annotations


from dotenv import load_dotenv
load_dotenv("/Users/edenhyacinth/autodocs/.env")

import json
import pathlib
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from autodocs.document.doc import OutputDocumentation
from autodocs.prompts.called_fn_summarisation.run import CalledFnSummarisationQA
from autodocs.prompts.condenser.run import CondenserQA
from autodocs.prompts.hyperparameters.run import HyperparameterQA
from autodocs.prompts.usefulness.run import ImportanceQA
from autodocs.utils.function_description import FunctionDescription
from autodocs.utils.slugify import slugify


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
                arguments: dict[str, str] = function_info["arguments"]
                source: str = function_info["source"]
                docs: str = function_info["caller_docs"]
                signature: Optional[str] = function_info["signature"]
                if source is not None and source != "None":
                    return FunctionDescription(function_name, source, docs, arguments, signature)
        except FileNotFoundError:
            return FunctionDescription(function_name, "", "", {})

    def _load_important_function_calls(self) -> list[str]:
        return ImportanceQA()(self.trace)

    def _retrieve_function_descriptions(
        self, important_functions: list[str]
    ) -> list[FunctionDescription]:
        return [
            self.load_function_info(function_call_name)
            for function_call_name in important_functions
        ]

    @staticmethod
    def _summarise_functions_with_arguments(
        function_descriptions: list[FunctionDescription]
    ) -> list[tuple[FunctionDescription, str]]:
        return list(
            CalledFnSummarisationQA()(functions=function_descriptions)
        )

    def _condense_descriptions(
        self, function_summaries: list[tuple[FunctionDescription, str]]
    ) -> str:
        return CondenserQA()(
            functions=function_summaries,
            trace=self.trace_fns,
            trace_type=self.trace_type,
        )

    def _retrieve_hyperparameters(
        self, function_summaries: list[tuple[FunctionDescription, str]]
    ) -> list[tuple[FunctionDescription, dict[str, Any]]]:
        return list(HyperparameterQA()(function_summaries))

    def create_description(self) -> str:
        # Identify usefulness of each component within the trace
        important_functions = self._load_important_function_calls()
        print(f"{important_functions=}")
        # Retrieve Source info for each of the important functions from the dump
        function_descriptions = self._retrieve_function_descriptions(
            important_functions
        )
        # Summarise the Function Source Info
        function_summaries = self._summarise_functions_with_arguments(
            function_descriptions
        )
        print(f"Fn Summaries: {[summary for (_, summary) in function_summaries]}")
        # Identify hyperparameters within the functions, based on the descriptions of the functions.
        #   TODO - Extend this using saved items - frequently it's not possible to identify the objects referred to because
        #       they existed on an object, not on the function. Having a saved item via Jackdaw would make this accessible again.
        hyperparameters = self._retrieve_hyperparameters(function_summaries)
        print(f"Hyperparameters: {[hp for (_, hp) in hyperparameters]}")
        condensed = self._condense_descriptions(function_summaries)
        breakpoint()


def load_library_versions(observation_id: uuid.UUID) -> dict[str, str]:
    version_path = root_path() / str(observation_id) / "versions.txt"
    if not version_path.exists():
        raise NoSuchTrace
    with open(version_path) as f:
        return json.load(f)


def create_output_document(observation_id: uuid.UUID) -> OutputDocumentation:
    try:
        processing_description = Trace.from_id(
            observation_id, "Processing"
        ).create_description()
    except NoSuchTrace:
        processing_description = None
    try:
        training_description = Trace.from_id(
            observation_id, "Training"
        ).create_description()
    except NoSuchTrace:
        training_description = None
    try:
        inference_description = Trace.from_id(
            observation_id, "Inference"
        ).create_description()
    except NoSuchTrace:
        inference_description = None

    return OutputDocumentation(
        preprocessing_steps=processing_description,
        training_steps=training_description,
        inference_steps=inference_description,
        total_libraries=load_library_versions(observation_id),
    )


if __name__ == "__main__":
    inference_trace = Trace.from_id(
        uuid.UUID("a171d4c6-ba7d-4594-8b98-54f8e41a1229"), trace_type="Inference"
    )
    print(inference_trace.create_description().steps())
