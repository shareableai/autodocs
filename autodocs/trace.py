from __future__ import annotations

import json
import logging
import pathlib
import uuid
from dataclasses import dataclass

from sentiml.tracking_type import TrackingType

from autodocs.hyperparameters import function_hyperparameters
from autodocs.prompts.filter_description.run import FilterQA
from autodocs.prompts.function_parameter_summariser.run import FnSummariserQA
from autodocs.prompts.trace_description.run import TraceQA
from autodocs.prompts.trace_graph.run import TraceGraphQA
from autodocs.utils.function_description import FunctionDescription

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def root_path() -> pathlib.Path:
    return pathlib.Path.home() / ".stack_traces"


class NoSuchTrace(BaseException):
    pass


@dataclass
class Trace:
    id: uuid.UUID
    root_dir: pathlib.Path
    trace: list[str]
    trace_fns: list[str]
    trace_type: str

    @staticmethod
    def from_id(trace_id: uuid.UUID, trace_type: TrackingType) -> Trace:
        directory = root_path() / str(trace_id) / str(trace_type)
        if not directory.exists():
            raise NoSuchTrace(f"Could not find {directory}")
        with open(directory / "trace.txt") as f:
            trace = f.readlines()
            trace_fns = [
                fn_desc[4:].replace("\t", "").replace(" ", "").replace("\n", "")
                for fn_desc in trace
            ]
        return Trace(trace_id, directory, trace, trace_fns, str(trace_type))

    def describe_trace(self) -> None:
        trace_path = (
                pathlib.Path.home()
                / ".autodocs"
                / str(self.id)
                / self.trace_type
                / f"trace_description.txt"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as trace_path_writable:
            trace_path_writable.write(TraceQA()("\n".join(self.trace), self.trace_type))

    def trace_graph(self) -> None:
        trace_path = (
                pathlib.Path.home()
                / ".autodocs"
                / str(self.id)
                / self.trace_type
                / f"trace_graph.txt"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as trace_path_writable:
            trace_path_writable.write(
                TraceGraphQA()("\n".join(self.trace), self.trace_type)
            )

    @staticmethod
    def _format_class_name(class_dict: dict) -> str:
        return '.'.join(class_dict['metadata']['module']) + "." + class_dict['metadata']['name']

    def describe_classes(self) -> None:
        trace_path = (
                pathlib.Path.home()
                / ".autodocs"
                / str(self.id)
                / self.trace_type
                / f"classes.json"
        )
        class_dir = self.root_dir.parent / 'classes'
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as trace_path_writable:
            json.dump(
                {
                    class_id: self._format_class_name(class_dict) for (class_id, class_dict) in ([
                    (class_file.stem, json.load(open(class_file)))
                        for class_file
                        in class_dir.glob('*.json')
                ])
                },
                trace_path_writable
            )

    def save_hyperparameters(self) -> None:
        trace_path = (
                pathlib.Path.home()
                / ".autodocs"
                / str(self.id)
                / self.trace_type
                / "hyperparameters"
        )
        trace_path.mkdir(parents=True, exist_ok=True)
        for function_name in self.root_dir.glob("*"):
            if function_name.name == "trace.txt":
                continue
            fn_item = json.load(open(function_name, "r"))
            if len(fn_item["tracked_argument_ids"]) > 0:
                parameter_description = {}
                fn_desc = FunctionDescription.from_file(
                    self.root_dir, function_name.stem
                )
                hyperparameters = function_hyperparameters(fn_desc)
                parameter_summary = FnSummariserQA()(fn_desc)
                logging.info(f"Parameter Summary: {parameter_summary}")
                for param in hyperparameters.keys():
                    parameter_description[param] = FilterQA()(
                        parameter_summary,
                        f'exclusively the description of the parameter {param}',
                        'a short sentence that keeps the original meaning of the description',
                        ""
                    )
                with open(trace_path / f"{fn_desc.name}.json", "w") as function_path:
                    tracked_ids = fn_item["tracked_argument_ids"]
                    json.dump(
                        {
                            "class_id": tracked_ids.get(
                                "self", tracked_ids.get("cls", None)
                            ),
                            "hyperparameters": hyperparameters,
                            "descriptions": parameter_description
                        },
                        function_path,
                    )


if __name__ == "__main__":
    trace = Trace.from_id(
        uuid.UUID("47909c92-8455-4734-a35f-0a0f1588dd17"), TrackingType.Training
    )
    trace.save_hyperparameters()
    trace.describe_trace()
    trace.trace_graph()
    trace.describe_classes()
