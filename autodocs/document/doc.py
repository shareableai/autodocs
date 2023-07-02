import pathlib
from dataclasses import dataclass, field
from typing import *

from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader("autodocs"),
    autoescape=select_autoescape()
)

model_html_template = env.get_template("model_page.html")

@dataclass
class Parameter:
    name: str
    value: str


@dataclass
class Description:
    component_name: str
    component_description: str

@dataclass
class OutputDocumentation:
    workspace_name: Optional[str] = None
    preprocessing_steps: dict[str, str] = field(default_factory=dict)
    training_steps: dict[str, str] = field(default_factory=dict)
    inference_steps: dict[str, str] = field(default_factory=dict)
    traced_libraries: Dict[str, str] = field(default_factory=dict)
    total_libraries: Dict[str, str] = field(default_factory=dict)
    preprocessing_hyperparams: Dict[str, Any] = field(default_factory=dict)
    training_hyperparams: Dict[str, Any] = field(default_factory=dict)
    inference_hyperparams: Dict[str, Any] = field(default_factory=dict)

    def steps(self) -> dict[str, str]:
        return {
            "preprocessing": self.preprocessing_steps,
            "training": self.training_steps,
            "inference": self.inference_steps,
        }
    
    def to_html(self, output_directory: pathlib.Path) -> None:
        if self.workspace_name is not None:
            output_dir: pathlib.Path = output_directory / self.workspace_name
        else:
            output_dir = output_directory
        output_dir.mkdir(exist_ok=True, parents=True)
        for section in ['preprocessing', 'training', 'inference']:
            with open(output_dir / f"{section}.html", 'w') as results:
                rendered_page = model_html_template.render(
                    {
                        "model_name": self.workspace_name if self.workspace_name is not None else "Unnamed Model",
                        "descriptions": [Description(k, v) for (k, v) in getattr(self, f"{section}_steps").items()],
                        "parameters": [Parameter(k, str(v)) for (k,v) in getattr(self, f"{section}_hyperparams", {}).items()]
                    }
                )
                results.write(rendered_page)
