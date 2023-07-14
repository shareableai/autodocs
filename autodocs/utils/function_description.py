from __future__ import annotations

import pathlib
import json
import logging

from dataclasses import dataclass
from typing import Optional, Any
from uuid import UUID
import uuid

from weaver.data import read_json_dict, WovenClass, ArtefactID
from weaver.unweave import unweave
from weaver.artefact_registry import ArtefactRegistry

from autodocs.utils.slugify import slugify

from autodocs.utils.functional import skip

LOGGER = logging.getLogger(__name__)


def load_artefact_from_file(id: ArtefactID) -> Any:
    return ArtefactRegistry().load_from_id(id)


def load_tracked_class_property(
    class_name: UUID, trace_id: str, name: str
) -> Optional[Any]:
    class_path = (
        pathlib.Path.home()
        / ".stack_traces"
        / trace_id
        / "classes"
        / f"{class_name}.json"
    )
    if not class_path.exists():
        print(f"No class path found - {class_path=}")
        return None
    with open(class_path) as f:
        class_item = read_json_dict(json.load(f))
        if name.startswith("self") or name.startswith("cls"):
            sub_items = skip(name.split("."), 1)
        else:
            sub_items = name.split(".")
        for sub_item in sub_items:
            try:
                class_item = class_item.json[sub_item]
            except KeyError:
                LOGGER.warning(f"Could not find {name} on class")
                return None
        if (
            isinstance(class_item, WovenClass)
            and class_item.metadata.name == "ArtefactID"
        ):
            return load_artefact_from_file(unweave(class_item))
        else:
            return class_item


@dataclass
class FunctionDescription:
    name: str
    source: str
    docs: str
    arguments: dict[str, str]
    signature: Optional[str]
    root_dir: pathlib.Path
    caller_name: Optional[str]
    caller_docs: Optional[str]
    tracked_argument_ids: dict[str, UUID]

    def __hash__(self) -> int:
        return hash(
            tuple([self.name, self.source, self.docs, self.root_dir, self.caller_name])
        )

    @staticmethod
    def from_file(directory: pathlib.Path, function_name: str) -> FunctionDescription:
        try:
            with open(directory / slugify(function_name)) as f:
                function_info = json.load(f)
                arguments: dict[str, str] = function_info.get("arguments", {})
                source: str = function_info.get("source", "")
                docs: str = function_info.get("caller_docs", "")
                caller_name: str = function_info.get("caller_name", "")
                caller_docs: str = function_info.get("caller_docs", "")
                signature: Optional[str] = function_info.get("signature", None)
                tracked_argument_ids: dict[str, uuid.UUID] = {
                    k: uuid.UUID(v)
                    for (k, v) in function_info.get(
                        "tracked_argument_ids", dict()
                    ).items()
                }
                return FunctionDescription(
                    name=function_name,
                    source=source,
                    docs=docs,
                    arguments=arguments,
                    signature=signature,
                    root_dir=directory,
                    caller_name=caller_name,
                    caller_docs=caller_docs,
                    tracked_argument_ids=tracked_argument_ids,
                )
        except FileNotFoundError:
            return FunctionDescription(
                function_name, "", "", {}, None, directory, None, {}
            )

    def retrieve_property(self, name: str) -> Optional[Any]:
        if name.startswith("self.") or name.startswith("cls."):
            initial_argument = next(iter(name.split(".", 1)))
            if (
                class_name := self.tracked_argument_ids.get(initial_argument, None)
            ) is not None:
                return load_tracked_class_property(
                    class_name, self.root_dir.parent.name, name
                )
            else:
                return None
        else:
            return self.arguments.get(name, None)
