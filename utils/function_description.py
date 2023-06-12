from dataclasses import dataclass


@dataclass
class FunctionDescription:
    name: str
    source: str
    docs: str
    arguments: dict[str, str]
