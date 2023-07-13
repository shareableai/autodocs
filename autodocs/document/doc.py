from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import *

from jinja2 import Environment, PackageLoader, select_autoescape
from observer.tracking_type import TrackingType

env = Environment(loader=PackageLoader("autodocs"), autoescape=select_autoescape())

model_html_template = env.get_template("model_page.html")


@dataclass
class Parameter:
    name: str
    value: str


@dataclass
class Parameter:
    name: str
    value: str
    reasoning: str


@dataclass
class TrackedClassDescription:
    class_name: str
    parameters: list[Parameter] = field(default_factory=list)


@dataclass
class TrackedFunctionParameters:
    function_name: str
    parameters: list[Parameter] = field(default_factory=list)


@dataclass
class OutputSection:
    section_name: str
    trace_description: str
    tracked_class_parameters: list[TrackedClassDescription] = field(
        default_factory=list
    )
    function_hyperparameters: list[TrackedFunctionParameters] = field(
        default_factory=list
    )
    trace_graph: list[str] = field(default_factory=list)

    @staticmethod
    def generate_section(
        section: TrackingType, section_path: pathlib.Path
    ) -> OutputSection:
        trace_description = open(section_path / "trace_description.txt", "r").read()
        trace_description_graph = open(
            section_path / "trace_graph.txt", "r"
        ).readlines()
        tracked_class_parameters: dict[str, list[Parameter]] = {}
        free_function_parameters = []
        with open(section_path / 'classes.json') as classes_file:
            class_lookup = json.load(classes_file)
            for tracked_function_file in (section_path / "hyperparameters").glob("*.json"):
                with open(tracked_function_file) as tracked_function_file_readable:
                    tracked_function = json.load(tracked_function_file_readable)
                    class_parameter_values = {}
                    class_parameter_reasoning = {}
                    floating_parameter_values = {}
                    floating_parameter_reasoning = {}
                    descriptions: dict[str, str] = tracked_function['descriptions']
                    for parameter, value_description in tracked_function[
                        "hyperparameters"
                    ].items():
                        if parameter.startswith("self") or parameter.startswith("cls"):
                            class_parameter_values[parameter] = value_description[0]
                            class_parameter_reasoning[parameter] = descriptions.get(parameter, value_description[1])
                        else:
                            floating_parameter_values[parameter] = value_description[0]
                            floating_parameter_reasoning[parameter] = descriptions.get(parameter, value_description[1])
                    if len(tracked_function["hyperparameters"]) > 0:
                        if len(class_parameter_values) > 0:
                            class_id = tracked_function["class_id"]
                            class_name = class_lookup.get(class_id, None)
                            class_parameters = [
                                Parameter(
                                    parameter_name,
                                    class_parameter_values[parameter_name],
                                    class_parameter_reasoning[parameter_name],
                                )
                                for parameter_name in class_parameter_values.keys()
                            ]
                            try:
                                tracked_class_parameters[class_name] = (
                                    tracked_class_parameters[class_name] + class_parameters
                                )
                            except KeyError:
                                tracked_class_parameters[class_name] = class_parameters
                        if len(floating_parameter_values) > 0:
                            floating_parameters = [
                                Parameter(
                                    parameter_name,
                                    floating_parameter_values[parameter_name],
                                    floating_parameter_reasoning[parameter_name],
                                )
                                for parameter_name in floating_parameter_values.keys()
                            ]
                            free_function_parameters.append(
                                TrackedFunctionParameters(
                                    tracked_function_file.stem, floating_parameters
                                )
                            )
        return OutputSection(
            section.name,
            trace_description,
            tracked_class_parameters=[
                TrackedClassDescription(class_name, parameter_list)
                for (class_name, parameter_list) in tracked_class_parameters.items()
            ],
            function_hyperparameters=free_function_parameters,
            trace_graph=trace_description_graph,
        )


@dataclass
class OutputDocumentation:
    workspace_name: Optional[str] = None
    sections: List[OutputSection] = field(default_factory=list)

    @staticmethod
    def from_autodocs_directory(path: pathlib.Path) -> OutputDocumentation:
        sections = []
        for section_name in [TrackingType.Training, TrackingType.Inference]:
            try:
                sections.append(
                    OutputSection.generate_section(
                        section_name, path / str(section_name)
                    )
                )
            except FileNotFoundError:
                continue
        return OutputDocumentation(
            workspace_name=None,
            sections=sections,
        )

    def to_html(self, output_directory: pathlib.Path) -> None:
        if self.workspace_name is not None:
            output_dir: pathlib.Path = output_directory / self.workspace_name
        else:
            output_dir = output_directory
        output_dir.mkdir(exist_ok=True, parents=True)
        for section in self.sections:
            with open(output_dir / f"{section.section_name}.html", "w") as results:
                rendered_page = model_html_template.render(
                    {
                        "model_name": self.workspace_name
                        if self.workspace_name is not None
                        else "Unnamed Model",
                        "graph": section.trace_graph,
                        "description": section.trace_description,
                        "classes": section.tracked_class_parameters,
                        "functions": section.function_hyperparameters,
                    }
                )
                results.write(rendered_page)
