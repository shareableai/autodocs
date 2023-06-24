from dataclasses import dataclass, field
from typing import *


@dataclass
class OutputDocumentation:
    workspace_name: Optional[str] = None
    preprocessing_steps: Optional[str] = None
    training_steps: Optional[str] = None
    inference_steps: Optional[str] = None
    traced_libraries: Dict[str, str] = field(default_factory=dict)
    total_libraries: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def steps(self) -> dict[str, str]:
        return {
            "preprocessing": self.preprocessing_steps,
            "training": self.training_steps,
            "inference": self.inference_steps,
        }
