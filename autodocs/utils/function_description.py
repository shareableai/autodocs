import pathlib
import json
import logging

from dataclasses import dataclass
from typing import Optional, Any
from uuid import UUID

from weaver.data import read_json_dict, WovenClass, ArtefactID
from weaver.unweave import unweave
from weaver.artefact_registry import ArtefactRegistry

LOGGER = logging.getLogger(__name__)

def load_artefact_from_file(id: ArtefactID) -> Any:
    return ArtefactRegistry.load_from_id(id)

@dataclass
class FunctionDescription:
    name: str
    source: str
    docs: str
    arguments: dict[str, str]
    signature: Optional[str]
    root_dir: pathlib.Path
    caller_name: Optional[str]
    tracked_class_name: Optional[UUID]

    def load_tracked_class_property(self, name: str) -> Optional[Any]:
        class_path = self.root_dir.parent / 'classes' / f"{self.tracked_class_name}.json"
        if not class_path.exists():
            return None
        with open(class_path) as f:
            class_item = read_json_dict(json.load(f))
            for sub_item in name.split('.'):
                try:
                    class_item = class_item.json[sub_item]
                except KeyError:
                    LOGGER.warn(f"Could not find {name} on class")
                    return None
            if isinstance(class_item, WovenClass) and class_item.name == 'ArtefactID':
                return load_artefact_from_file(unweave(class_item))
            else:
                return class_item
